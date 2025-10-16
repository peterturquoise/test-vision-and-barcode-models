#!/usr/bin/env python3
"""
ZXing-CPP Model Implementation
Fast barcode detection using ZXing-CPP library with proven approach
"""

import cv2
import numpy as np
from PIL import Image
import subprocess
import tempfile
import os
import logging
from typing import Dict, List, Optional, Tuple
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BarcodeDetection:
    """Barcode detection result"""
    def __init__(self, bbox, value, barcode_type, confidence):
        self.bbox = bbox
        self.value = value
        self.barcode_type = barcode_type
        self.confidence = confidence

class BarcodeProcessor:
    """
    Barcode detection processor focused on BPost business logic:
    1. Look for 3232 barcodes (BPost) for direct DB lookup
    2. If no 3232 found, collect all barcodes for further processing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.zxing_path = "/usr/local/bin/ZXingReader"
    
    def detect_bpost_packages_from_array(self, image: np.ndarray) -> Dict:
        """
        Detect barcodes in an image, with special focus on BPost 3232 barcodes
        
        Args:
            image: OpenCV image array
            
        Returns:
            Dict with detection results
        """
        try:
            self.logger.debug(f"Input image dimensions: {image.shape}")
            
            # Initial detection with basic approaches
            all_barcodes = []
            bpost_barcodes = []
            
            # Try different approaches for barcode detection
            approaches = [
                "Full Image Threshold",
                "Full Image Grayscale", 
                "Full Image Enhanced"
            ]
            
            for approach in approaches:
                self.logger.debug(f"Trying approach: {approach}")
                processed_image = self._preprocess_image(image, approach)
                detected = self._detect_barcodes(processed_image, approach)
                
                # Log what was found in this approach
                if detected:
                    self.logger.info(f"✅ {approach}: Found {len(detected)} barcodes")
                    for barcode in detected:
                        self.logger.info(f"  📋 Barcode: {barcode['data']} (Type: {barcode['type']}, Confidence: {barcode['confidence']})")
                else:
                    self.logger.info(f"❌ {approach}: No barcodes found")
                
                # Add to results
                all_barcodes.extend(detected)
                
                # Check for 3232 barcodes - EXIT EARLY if found!
                for barcode in detected:
                    if barcode['data'].startswith('3232'):
                        bpost_barcodes.append(barcode)
                        self.logger.info(f"🎯 Found BPost barcode: {barcode['data']} via {approach}")
                        # ✅ EXIT EARLY - we found what we need!
                        self.logger.info("🚀 Early exit: BPost barcode found, skipping remaining approaches")
                        break
                
                # Exit early if we found BPost barcodes
                if bpost_barcodes:
                    break
            
            # Log summary of basic scan results
            self.logger.info(f"📊 Basic scan summary: {len(all_barcodes)} total barcodes found, {len(bpost_barcodes)} BPost barcodes")
            if all_barcodes:
                self.logger.info("📋 All barcodes found in basic scan:")
                for barcode in all_barcodes:
                    self.logger.info(f"  • {barcode['data']} (Type: {barcode['type']}, Approach: {barcode['approach']})")
            
            # If no 3232 barcodes found, try comprehensive fallback
            if not bpost_barcodes:
                self.logger.info("🔍 No 3232 barcodes found in basic scan, trying comprehensive fallback...")
                fallback_results = self._comprehensive_fallback(image)
                
                # Add fallback barcodes to results
                all_barcodes.extend(fallback_results.get('all_barcodes', []))
                bpost_barcodes.extend(fallback_results.get('bpost_barcodes', []))
                
                # Include the fallback report in the results
                comprehensive_fallback_report = fallback_results.get('fallback_report', {})
            else:
                comprehensive_fallback_report = {}
            
            # Prepare results
            results = {
                "status": "success",
                "workflow": "barcode_detection",
                "total_barcodes": len(all_barcodes),
                "total_bpost_barcodes": len(bpost_barcodes),
                "all_barcodes": all_barcodes,
                "bpost_barcodes": bpost_barcodes,
                "processing_time": 0  # Will be filled by caller
            }
            
            # Add comprehensive fallback report if available
            if comprehensive_fallback_report:
                results["comprehensive_fallback_report"] = comprehensive_fallback_report
            
            # Final comprehensive summary
            self.logger.info(f"🎯 FINAL SUMMARY: {len(all_barcodes)} total barcodes detected, {len(bpost_barcodes)} BPost barcodes")
            if all_barcodes:
                self.logger.info("📋 Complete barcode inventory:")
                for i, barcode in enumerate(all_barcodes, 1):
                    bpost_indicator = "🎯 BPost" if barcode['data'].startswith('3232') else "📋 Other"
                    self.logger.info(f"  {i}. {bpost_indicator}: {barcode['data']} (Type: {barcode['type']}, Approach: {barcode['approach']})")
            else:
                self.logger.warning("⚠️ No barcodes detected in any approach")
                
            return results
            
        except Exception as e:
            self.logger.error(f"Barcode detection failed: {str(e)}")
            return {
                "status": "error",
                "workflow": "barcode_detection",
                "error": str(e)
            }
    
    def _preprocess_image(self, image: np.ndarray, approach: str) -> np.ndarray:
        """Preprocess image based on approach"""
        self.logger.debug(f"Preprocessing image: shape={image.shape}, dtype={image.dtype}, approach={approach}")
        
        if approach == "Full Image Threshold":
            # Simple binary threshold
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image  # Already grayscale
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            return binary
            
        elif approach == "Full Image Grayscale":
            # Just convert to grayscale
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                return image  # Already grayscale
            
        elif approach == "Full Image Enhanced":
            # Enhance contrast
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image  # Already grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
            
        # Default: return original image
        return image
    
    def _detect_barcodes(self, image: np.ndarray, approach: str) -> List[Dict]:
        """Detect barcodes in an image using ZXingReader executable"""
        try:
            # Debug: Log image properties
            self.logger.debug(f"🔍 {approach} - Image shape: {image.shape}, dtype: {image.dtype}")
            self.logger.debug(f"🔍 {approach} - Image min/max: {image.min()}/{image.max()}")
            
            # Ensure image is uint8
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
                self.logger.debug(f"🔍 {approach} - Converted to uint8")
            
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, image)
            
            try:
                # Run ZXingReader on the image
                result = subprocess.run([
                    self.zxing_path,
                    temp_path
                ], capture_output=True, text=True, timeout=10)
                
                barcode_detections = []
                
                if result.returncode == 0 and result.stdout.strip():
                    # Parse output - ZXingReader outputs one barcode per line
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            # Extract barcode data
                            barcode_data = line.strip()
                            
                            # Try to determine barcode type based on length and format
                            barcode_type = self._guess_barcode_type(barcode_data)
                            
                            # Calculate confidence (zxing-cpp doesn't provide confidence)
                            confidence = 0.95 if len(barcode_data) > 0 else 0.5
                            
                            # Debug: Log all barcodes with type information
                            self.logger.info(f"🔍 Barcode detected: {barcode_data} (Type: {barcode_type})")
                            
                            barcode_detections.append({
                                'data': barcode_data,
                                'type': barcode_type,
                                'bbox': {'x': 0, 'y': 0, 'width': 0, 'height': 0},  # ZXingReader doesn't provide bbox
                                'approach': approach,
                                'confidence': confidence,
                                'source': 'full_image_scan'
                            })
                
                return barcode_detections
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            self.logger.error(f"Barcode detection failed: {str(e)}")
            return []
    
    def _guess_barcode_type(self, barcode_data: str) -> str:
        """Guess barcode type based on data format"""
        if len(barcode_data) == 24 and barcode_data.isdigit():
            return "CODE128"  # Likely a CODE128 barcode
        elif len(barcode_data) == 13 and barcode_data.isdigit():
            return "EAN13"
        elif len(barcode_data) == 12 and barcode_data.isdigit():
            return "UPC12"
        elif len(barcode_data) == 8 and barcode_data.isdigit():
            return "EAN8"
        elif barcode_data.startswith("http"):
            return "QR_CODE"
        else:
            return "UNKNOWN"
    
    def _comprehensive_fallback(self, image: np.ndarray) -> Dict:
        """
        Comprehensive fallback for difficult barcode detection cases
        """
        self.logger.info("🔍 COMPREHENSIVE FALLBACK: No 3232 barcodes found, trying advanced approaches...")
        
        all_barcodes = []
        bpost_barcodes = []
        fallback_report = {
            "approaches_tried": [],
            "successful_approaches": [],
            "total_barcodes_found": 0,
            "total_bpost_barcodes_found": 0
        }
        
        # Define zxing-cpp optimized approaches based on diagnostic testing
        advanced_approaches = [
            # PROVEN SUCCESSFUL APPROACHES (from diagnostic testing) - try these first!
            ("Higher Contrast", lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0)),  # ✅ FOUND BPOST BARCODE
            ("Bilateral Filter", lambda img: cv2.bilateralFilter(img, 9, 75, 75)),  # ✅ FOUND BPOST BARCODE
            
            # Scale approaches (zxing-cpp benefits from higher resolution, but skip 3x for performance)
            ("Scale 1.5x", lambda img: cv2.resize(img, (int(img.shape[1]*1.5), int(img.shape[0]*1.5)))),
            ("Scale 2x", lambda img: cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))),
            
            # Additional contrast enhancement approaches
            ("Higher Contrast 2x", lambda img: cv2.convertScaleAbs(img, alpha=2.0, beta=0)),
            ("Higher Contrast + Bright", lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=20)),
            ("CLAHE Enhancement", lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))),
            
            # Original image and basic preprocessing
            ("Original Image", lambda img: img),
            ("Grayscale", lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
            
            # Noise reduction approaches
            ("Light Gaussian Blur", lambda img: cv2.GaussianBlur(img, (3, 3), 0)),
            ("Medium Gaussian Blur", lambda img: cv2.GaussianBlur(img, (5, 5), 0)),
            
            # Small rotation adjustments - TESTING: Commented out to test if ZXingReader handles small angles automatically
            # ("Rotate -1°", lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), -1, 1), (img.shape[1], img.shape[0]))),
            # ("Rotate +1°", lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 1, 1), (img.shape[1], img.shape[0]))),
            # ("Rotate -2°", lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), -2, 1), (img.shape[1], img.shape[0]))),
            # ("Rotate +2°", lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 2, 1), (img.shape[1], img.shape[0]))),
            
            # Additional preprocessing approaches
            ("Edge Preserving Filter", lambda img: cv2.edgePreservingFilter(img, flags=1, sigma_s=50, sigma_r=0.4)),
            ("Adaptive Threshold", lambda img: cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        ]
        
        # Try each approach
        for i, (name, process_func) in enumerate(advanced_approaches):
            fallback_report["approaches_tried"].append(name)
            self.logger.info(f"🔍 Fallback attempt {i+1}/{len(advanced_approaches)}: {name}")
            
            try:
                # Process image
                processed = process_func(image)
                
                # Detect barcodes using the same method as basic approaches
                detected = self._detect_barcodes(processed, f"Fallback: {name}")
                
                if detected:
                    fallback_report["successful_approaches"].append(name)
                    
                    # Process results
                    for barcode in detected:
                        # Create result
                        result = {
                            'data': barcode['data'],
                            'type': barcode['type'],
                            'bbox': barcode['bbox'],
                            'approach': f"Fallback: {name}",
                            'confidence': 0.6,  # Lower confidence for fallback
                            'source': 'fallback_scan'
                        }
                        
                        # Add to results
                        all_barcodes.append(result)
                        
                        # Check for 3232 barcodes - EXIT EARLY if found!
                        if barcode['data'].startswith('3232'):
                            bpost_barcodes.append(result)
                            self.logger.info(f"🎯 Found BPost barcode: {barcode['data']} via fallback {name}")
                            # ✅ EXIT EARLY - we found what we need!
                            self.logger.info("🚀 Early exit: BPost barcode found in fallback, skipping remaining approaches")
                            break
                        else:
                            self.logger.info(f"  📋 Barcode: {barcode['data']} (Type: {barcode['type']})")
                            
                    self.logger.info(f"✅ {name}: Found {len(detected)} barcodes")
                    
                    # Exit early if we found BPost barcodes
                    if bpost_barcodes:
                        break
                else:
                    self.logger.info(f"❌ {name}: No barcodes found")
                    
            except Exception as e:
                self.logger.error(f"❌ {name} failed: {str(e)}")
        
        # Update fallback report
        fallback_report["total_barcodes_found"] = len(all_barcodes)
        fallback_report["total_bpost_barcodes_found"] = len(bpost_barcodes)
        
        self.logger.info(f"🔍 Fallback complete: {len(all_barcodes)} total barcodes, {len(bpost_barcodes)} BPost barcodes")
        
        # Log detailed summary of all fallback barcodes
        if all_barcodes:
            self.logger.info("📋 All barcodes found in fallback:")
            for barcode in all_barcodes:
                self.logger.info(f"  • {barcode['data']} (Type: {barcode['type']}, Approach: {barcode['approach']})")
        
        return {
            "all_barcodes": all_barcodes,
            "bpost_barcodes": bpost_barcodes,
            "fallback_report": fallback_report
        }

class ZXingCPPModel:
    """ZXing-CPP implementation for fast barcode detection using proven approach"""
    
    def __init__(self):
        self.device = "cpu"
        self.logger = logging.getLogger(__name__)
        self.processor = BarcodeProcessor()
        self.load_model()
    
    def load_model(self):
        """Load ZXing-CPP model"""
        try:
            # Test zxing-cpp executable
            result = subprocess.run([self.processor.zxing_path, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("ZXing-CPP executable loaded successfully")
                logger.info(f"Version: {result.stdout.strip()}")
            else:
                logger.error(f"Error testing zxing-cpp: {result.stderr}")
                raise Exception("ZXing-CPP executable not working")
        except Exception as e:
            logger.error(f"Error loading ZXing-CPP: {e}")
            raise
    
    def detect_barcodes(self, image):
        """Detect barcodes in the image using the proven approach"""
        try:
            # Convert to BGR numpy array as expected by BarcodeProcessor
            if isinstance(image, Image.Image):
                # PIL Image -> RGB numpy array -> BGR numpy array
                image_np = np.array(image)  # Shape: (H, W, 3), dtype: uint8, format: RGB
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
            else:
                # Assume it's already a numpy array
                image_np = image
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    # Check if it's RGB and convert to BGR
                    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    image_cv = image_np  # Already in correct format
            
            # Ensure the image is in the correct format for BarcodeProcessor
            # Expected: numpy.ndarray, shape (H, W, 3), dtype uint8, BGR format
            if image_cv.dtype != np.uint8:
                image_cv = image_cv.astype(np.uint8)
            
            self.logger.debug(f"Image ready for BarcodeProcessor: shape={image_cv.shape}, dtype={image_cv.dtype}")
            
            # Use the proven approach
            start_time = time.time()
            results = self.processor.detect_bpost_packages_from_array(image_cv)
            processing_time = time.time() - start_time
            
            # Convert results to BarcodeDetection objects
            barcode_detections = []
            for barcode in results.get('all_barcodes', []):
                bbox = [
                    barcode['bbox']['x'],
                    barcode['bbox']['y'], 
                    barcode['bbox']['width'],
                    barcode['bbox']['height']
                ]
                barcode_detections.append(BarcodeDetection(
                    bbox=bbox,
                    value=barcode['data'],
                    barcode_type=barcode['type'],
                    confidence=barcode['confidence']
                ))
            
            return barcode_detections
            
        except Exception as e:
            logger.error(f"Error detecting barcodes: {str(e)}")
            return []
    
    def describe_image(self, image):
        """Describe the image content (not applicable for barcode-only model)"""
        return "ZXing-CPP is a barcode-only detection model. Use detect_barcodes() for barcode detection."
    
    def extract_text(self, image):
        """Extract text from image (not applicable for barcode-only model)"""
        return []