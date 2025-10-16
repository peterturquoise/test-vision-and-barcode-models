#!/usr/bin/env python3
"""
ZXing-CPP Model with Parallel Preprocessing Approaches
Uses multiprocessing to run preprocessing approaches in parallel
"""

import subprocess
import tempfile
import time
import re
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import threading

class ZXingCPPModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.zxing_path = "/usr/local/bin/ZXingReader"
        
        # Define preprocessing approaches
        self.approaches = [
            "Original",
            "Grayscale", 
            "Higher Contrast",
            "Scale 1.5x",
            "Scale 2.0x",
            "Threshold Binary",
            "Threshold Adaptive",
            "Morphological Opening",
            "Gaussian Blur",
            "Sharpening"
        ]
        
        # Thread lock for logging
        self.log_lock = threading.Lock()
        
    def detect_barcodes(self, image):
        """
        Detect barcodes using parallel preprocessing approaches
        Returns list of barcode dictionaries with proper parsing
        """
        try:
            # Convert PIL Image to BGR NumPy array (OpenCV format)
            if isinstance(image, Image.Image):
                # Convert PIL to numpy array
                img_array = np.array(image)
                # Convert RGB to BGR for OpenCV
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_array = image
            
            self.logger.info(f"🔍 Starting parallel barcode detection with {len(self.approaches)} approaches")
            
            # Run approaches in parallel
            start_time = time.time()
            all_barcodes = self._run_parallel_approaches(img_array)
            total_time = time.time() - start_time
            
            # Deduplicate results
            unique_barcodes = self._deduplicate_barcodes(all_barcodes)
            
            self.logger.info(f"🔍 Found {len(unique_barcodes)} unique barcodes in {total_time:.3f}s using parallel processing")
            return unique_barcodes
            
        except Exception as e:
            self.logger.error(f"❌ Error in detect_barcodes: {e}")
            return []
    
    def _run_parallel_approaches(self, image: np.ndarray) -> List[Dict]:
        """Run all preprocessing approaches in parallel"""
        all_barcodes = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all approaches
            future_to_approach = {
                executor.submit(self._detect_with_approach, image, approach): approach 
                for approach in self.approaches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_approach):
                approach = future_to_approach[future]
                try:
                    barcodes = future.result()
                    if barcodes:
                        with self.log_lock:
                            self.logger.info(f"✅ {approach}: Found {len(barcodes)} barcodes")
                        all_barcodes.extend(barcodes)
                    else:
                        with self.log_lock:
                            self.logger.info(f"❌ {approach}: No barcodes found")
                except Exception as e:
                    with self.log_lock:
                        self.logger.error(f"❌ {approach}: Error - {e}")
        
        return all_barcodes
    
    def _detect_with_approach(self, image: np.ndarray, approach: str) -> List[Dict]:
        """Detect barcodes using a specific preprocessing approach"""
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image, approach)
            
            # Save processed image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, processed_image)
            
            try:
                # Run ZXingReader with -errors flag for maximum detection
                result = subprocess.run([
                    self.zxing_path, "-errors", temp_path
                ], capture_output=True, text=True, timeout=10)
                
                # Parse the output correctly
                barcodes = self._parse_zxing_output(result.stdout, approach)
                
                return barcodes
                
            finally:
                # Clean up temporary file
                Path(temp_path).unlink()
                
        except Exception as e:
            self.logger.error(f"❌ Error in {approach}: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray, approach: str) -> np.ndarray:
        """Preprocess image based on approach"""
        try:
            if approach == "Original":
                return image
                
            elif approach == "Grayscale":
                if len(image.shape) == 3:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return image
                
            elif approach == "Higher Contrast":
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                return clahe.apply(gray)
                
            elif approach == "Scale 1.5x":
                height, width = image.shape[:2]
                new_width = int(width * 1.5)
                new_height = int(height * 1.5)
                return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
            elif approach == "Scale 2.0x":
                height, width = image.shape[:2]
                new_width = int(width * 2.0)
                new_height = int(height * 2.0)
                return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
            elif approach == "Threshold Binary":
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
                return binary
                
            elif approach == "Threshold Adaptive":
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                
            elif approach == "Morphological Opening":
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
                
            elif approach == "Gaussian Blur":
                return cv2.GaussianBlur(image, (5, 5), 0)
                
            elif approach == "Sharpening":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                return cv2.filter2D(image, -1, kernel)
                
            else:
                return image
                
        except Exception as e:
            self.logger.error(f"❌ Preprocessing error in {approach}: {e}")
            return image
    
    def _parse_zxing_output(self, output: str, approach: str) -> List[Dict]:
        """Parse ZXingReader multi-line output correctly"""
        barcodes = []
        seen_barcodes = set()
        
        if not output.strip():
            return barcodes
        
        # Split output into blocks (each barcode is separated by empty lines)
        blocks = output.strip().split('\n\n')
        
        for block in blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            barcode_data = {}
            
            # Parse each line in the block
            for line in lines:
                line = line.strip()
                
                if line.startswith('Text:'):
                    # Extract the quoted text
                    match = re.search(r'"([^"]+)"', line)
                    if match:
                        barcode_data['value'] = match.group(1)
                
                elif line.startswith('Format:'):
                    # Extract format
                    format_match = re.search(r'Format:\s+(\w+)', line)
                    if format_match:
                        barcode_data['barcode_type'] = format_match.group(1)
                
                elif line.startswith('Position:'):
                    # Extract position
                    pos_match = re.search(r'Position:\s+(\d+)x(\d+)\s+(\d+)x(\d+)\s+(\d+)x(\d+)\s+(\d+)x(\d+)', line)
                    if pos_match:
                        # Convert to [x1, y1, x2, y2] format
                        x1, y1 = int(pos_match.group(1)), int(pos_match.group(2))
                        x2, y2 = int(pos_match.group(3)), int(pos_match.group(4))
                        x3, y3 = int(pos_match.group(5)), int(pos_match.group(6))
                        x4, y4 = int(pos_match.group(7)), int(pos_match.group(8))
                        
                        # Calculate bounding box
                        min_x = min(x1, x2, x3, x4)
                        max_x = max(x1, x2, x3, x4)
                        min_y = min(y1, y2, y3, y4)
                        max_y = max(y1, y2, y3, y4)
                        
                        barcode_data['bbox'] = [min_x, min_y, max_x, max_y]
                
                elif line.startswith('Rotation:'):
                    # Extract rotation
                    rot_match = re.search(r'Rotation:\s+(-?\d+)\s+deg', line)
                    if rot_match:
                        barcode_data['rotation'] = int(rot_match.group(1))
                
                elif line.startswith('Error:'):
                    # Extract error information
                    barcode_data['has_error'] = True
                    barcode_data['error_type'] = line.replace('Error:', '').strip()
            
            # Only add barcode if we have a value and it's not a duplicate
            if 'value' in barcode_data and barcode_data['value'] not in seen_barcodes:
                seen_barcodes.add(barcode_data['value'])
                
                # Set defaults for missing fields
                barcode_data.setdefault('barcode_type', 'UNKNOWN')
                barcode_data.setdefault('bbox', [0, 0, 0, 0])
                barcode_data.setdefault('rotation', 0)
                barcode_data.setdefault('has_error', False)
                barcode_data.setdefault('error_type', None)
                barcode_data.setdefault('confidence', 0.95 if not barcode_data['has_error'] else 0.8)
                barcode_data.setdefault('approach', approach)
                
                barcodes.append(barcode_data)
        
        return barcodes
    
    def _deduplicate_barcodes(self, all_barcodes: List[Dict]) -> List[Dict]:
        """Remove duplicate barcodes, keeping the best version of each"""
        unique_barcodes = {}
        
        for barcode in all_barcodes:
            value = barcode['value']
            
            if value not in unique_barcodes:
                # First occurrence
                unique_barcodes[value] = barcode
            else:
                # Duplicate - keep the one with higher confidence or no errors
                existing = unique_barcodes[value]
                
                # Prefer barcodes without errors
                if not barcode.get('has_error', False) and existing.get('has_error', False):
                    unique_barcodes[value] = barcode
                # If both have same error status, prefer higher confidence
                elif barcode.get('confidence', 0) > existing.get('confidence', 0):
                    unique_barcodes[value] = barcode
        
        return list(unique_barcodes.values())
