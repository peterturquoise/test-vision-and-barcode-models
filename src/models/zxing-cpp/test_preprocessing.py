#!/usr/bin/env python3
"""
Test Preprocessing Approaches
Re-test if different preprocessing approaches find different barcodes
"""

import subprocess
import tempfile
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def test_preprocessing_approaches(image_path):
    """Test different preprocessing approaches on the same image"""
    print(f"🔍 Testing preprocessing approaches on: {Path(image_path).name}")
    print("=" * 60)
    
    # Load image
    image = Image.open(image_path)
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Copy image to container
    subprocess.run([
        "docker", "cp", str(image_path), "zxing-pattern:/app/test_image.jpeg"
    ], check=True)
    
    # Define preprocessing approaches
    approaches = [
        ("Original", lambda img: img),
        ("Grayscale", lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img),
        ("Higher Contrast", lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0)),
        ("Scale 1.5x", lambda img: cv2.resize(img, (int(img.shape[1]*1.5), int(img.shape[0]*1.5)))),
        ("Scale 2.0x", lambda img: cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))),
        ("Threshold Binary", lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img, 128, 255, cv2.THRESH_BINARY)[1]),
        ("CLAHE Enhancement", lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img)),
    ]
    
    results = {}
    
    for approach_name, preprocess_func in approaches:
        print(f"\n📋 Testing: {approach_name}")
        print("-" * 40)
        
        try:
            # Preprocess image
            processed_img = preprocess_func(img_array)
            
            # Save processed image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, processed_img)
            
            # Copy to container
            subprocess.run([
                "docker", "cp", temp_path, "zxing-pattern:/app/processed_image.png"
            ], check=True)
            
            # Run ZXingReader
            result = subprocess.run([
                "docker", "exec", "zxing-pattern", 
                "/usr/local/bin/ZXingReader", "-errors", "/app/processed_image.png"
            ], capture_output=True, text=True, timeout=10)
            
            # Parse results
            barcodes = parse_zxing_output(result.stdout)
            
            print(f"✅ Found {len(barcodes)} barcodes:")
            for i, barcode in enumerate(barcodes, 1):
                print(f"  {i}. {barcode['value']} ({barcode['type']})")
            
            results[approach_name] = barcodes
            
            # Clean up
            Path(temp_path).unlink()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results[approach_name] = []
    
    # Compare results
    print(f"\n📊 COMPARISON:")
    print("=" * 60)
    
    all_barcodes = set()
    for approach, barcodes in results.items():
        for barcode in barcodes:
            all_barcodes.add(barcode['value'])
    
    print(f"Total unique barcodes found across all approaches: {len(all_barcodes)}")
    print("Unique barcodes:")
    for barcode in sorted(all_barcodes):
        print(f"  - {barcode}")
    
    # Check if approaches find different barcodes
    approach_barcodes = {}
    for approach, barcodes in results.items():
        approach_barcodes[approach] = set(barcode['value'] for barcode in barcodes)
    
    print(f"\n🔍 Do approaches find different barcodes?")
    print("-" * 40)
    
    all_same = True
    for approach1, barcodes1 in approach_barcodes.items():
        for approach2, barcodes2 in approach_barcodes.items():
            if approach1 != approach2 and barcodes1 != barcodes2:
                all_same = False
                print(f"❌ {approach1} ≠ {approach2}")
                print(f"   {approach1}: {sorted(barcodes1)}")
                print(f"   {approach2}: {sorted(barcodes2)}")
                break
        if not all_same:
            break
    
    if all_same:
        print("✅ All approaches find the same barcodes")
    else:
        print("❌ Different approaches find different barcodes!")
    
    return results

def parse_zxing_output(output):
    """Parse ZXingReader output"""
    barcodes = []
    seen_barcodes = set()
    
    if not output.strip():
        return barcodes
    
    blocks = output.strip().split('\n\n')
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        barcode_data = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Text:'):
                match = re.search(r'"([^"]+)"', line)
                if match:
                    barcode_data['value'] = match.group(1)
            
            elif line.startswith('Format:'):
                format_match = re.search(r'Format:\s+(\w+)', line)
                if format_match:
                    barcode_data['type'] = format_match.group(1)
        
        if 'value' in barcode_data and barcode_data['value'] not in seen_barcodes:
            seen_barcodes.add(barcode_data['value'])
            barcode_data.setdefault('type', 'UNKNOWN')
            barcodes.append(barcode_data)
    
    return barcodes

if __name__ == "__main__":
    import re
    import sys
    
    image_path = "/Users/peterbrooke/dev/cursor/test-7-vision-models/test_images/Complete Labels/3_J18CBEP8CCN070812400095N.jpeg"
    test_preprocessing_approaches(image_path)
