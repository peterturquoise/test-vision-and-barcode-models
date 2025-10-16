#!/usr/bin/env python3
"""
ZXing-CPP Model with Correct Parsing
Uses raw ZXingReader approach with proper multi-line parsing
"""

import subprocess
import tempfile
import time
import re
import logging
from pathlib import Path
import numpy as np
from PIL import Image

class ZXingCPPModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.zxing_path = "/usr/local/bin/ZXingReader"
        
    def detect_barcodes(self, image):
        """
        Detect barcodes using raw ZXingReader with correct parsing
        Returns list of barcode dictionaries with proper parsing
        """
        try:
            # Convert PIL Image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                image.save(temp_path, 'PNG')
            
            # Run ZXingReader directly (we're already inside the container)
            result = subprocess.run([
                self.zxing_path, "-errors", temp_path
            ], capture_output=True, text=True, timeout=30)
            
            # Clean up temporary file
            Path(temp_path).unlink()
            
            # Parse the output correctly
            barcodes = self._parse_zxing_output(result.stdout)
            
            self.logger.info(f"🔍 Found {len(barcodes)} barcodes using raw ZXingReader approach")
            return barcodes
            
        except Exception as e:
            self.logger.error(f"❌ Error in detect_barcodes: {e}")
            return []
    
    def _parse_zxing_output(self, output):
        """
        Parse ZXingReader multi-line output correctly
        Each barcode is represented by multiple lines:
        Text: "barcode_value"
        Format: Code39
        Position: x1 y1 x2 y2 x3 y3 x4 y4
        Rotation: 180 deg
        etc.
        """
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
                    # Extract the quoted text: "323201167200000687647030"
                    match = re.search(r'"([^"]+)"', line)
                    if match:
                        barcode_data['value'] = match.group(1)
                
                elif line.startswith('Format:'):
                    # Extract format: "DataMatrix", "Code39", etc.
                    format_match = re.search(r'Format:\s+(\w+)', line)
                    if format_match:
                        barcode_data['barcode_type'] = format_match.group(1)
                
                elif line.startswith('Position:'):
                    # Extract position: "1054x986 856x986 854x1010 1052x1010"
                    pos_match = re.search(r'Position:\s+(\d+)x(\d+)\s+(\d+)x(\d+)\s+(\d+)x(\d+)\s+(\d+)x(\d+)', line)
                    if pos_match:
                        # Convert to [x1, y1, x2, y2] format
                        x1, y1 = int(pos_match.group(1)), int(pos_match.group(2))
                        x2, y2 = int(pos_match.group(3)), int(pos_match.group(4))
                        x3, y3 = int(pos_match.group(5)), int(pos_match.group(6))
                        x4, y4 = int(pos_match.group(7)), int(pos_match.group(8))
                        
                        # Calculate bounding box (min/max coordinates)
                        min_x = min(x1, x2, x3, x4)
                        max_x = max(x1, x2, x3, x4)
                        min_y = min(y1, y2, y3, y4)
                        max_y = max(y1, y2, y3, y4)
                        
                        barcode_data['bbox'] = [min_x, min_y, max_x, max_y]
                
                elif line.startswith('Rotation:'):
                    # Extract rotation: "180 deg"
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
                
                barcodes.append(barcode_data)
                
                self.logger.info(f"🔍 Parsed barcode: {barcode_data['value']} ({barcode_data['barcode_type']})")
        
        return barcodes
