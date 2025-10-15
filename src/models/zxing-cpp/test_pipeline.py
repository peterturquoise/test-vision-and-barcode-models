#!/usr/bin/env python3
"""
Pipeline Test: Barcode Detection First, Then OCR Fallback
Test the complete pipeline with 1_J18CBEP8C7N070812400085N.jpeg
"""

import requests
import time
import json
from pathlib import Path

def test_pipeline_with_image(image_path):
    """
    Test the complete pipeline:
    1. Try barcode detection first (zxing-cpp)
    2. If barcodes found: Return barcode results (skip OCR)
    3. If no barcodes found: Fall back to OCR (PaddleOCR)
    """
    print(f"🔍 Testing Pipeline with: {image_path.name}")
    print("=" * 60)
    
    # Step 1: Try pattern-specific barcode detection first (BPost pattern)
    print("📋 STEP 1: BPost Barcode Detection (zxing-cpp pattern scanner)")
    print("-" * 40)
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8001/detect_barcode_pattern?pattern=3232", files=files, timeout=30)
        
        if response.status_code == 200:
            barcode_result = response.json()
            print(f"✅ Barcode detection successful!")
            print(f"📊 Processing time: {barcode_result.get('processing_time', 'N/A')}s")
            print(f"📋 Pattern found: {barcode_result.get('pattern_found', False)}")
            print(f"📋 Total barcodes found: {barcode_result.get('total_barcodes_found', 0)}")
            
            # Check if we found the BPost pattern
            pattern_found = barcode_result.get('pattern_found', False)
            barcodes = barcode_result.get('barcode_results', [])
            
            if pattern_found and barcodes:
                print(f"\n🎯 BPOST PATTERN DETECTED - Pipeline Complete!")
                print("📋 BPost barcode found:")
                
                for i, barcode in enumerate(barcodes, 1):
                    print(f"  {i}. {barcode.get('value', 'N/A')} ({barcode.get('barcode_type', 'N/A')})")
                
                print(f"\n✅ PIPELINE RESULT: BPost barcode found - skipping OCR step")
                return {
                    "success": True,
                    "method": "bpost_pattern_detection",
                    "pattern_found": True,
                    "bpost_barcode": barcodes[0].get('value', 'N/A'),
                    "processing_time": barcode_result.get('processing_time', 0),
                    "results": barcodes
                }
            else:
                print(f"\n❌ No BPost pattern (3232) detected - proceeding to OCR fallback")
        else:
            print(f"❌ Barcode detection failed: {response.status_code}")
            print(f"Response: {response.text}")
            return {"success": False, "error": "Barcode detection failed"}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return {"success": False, "error": str(e)}
    
    # Step 2: OCR Fallback (if no barcodes found)
    print(f"\n📝 STEP 2: OCR Fallback (PaddleOCR)")
    print("-" * 40)
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8000/extract_text", files=files, timeout=30)
        
        if response.status_code == 200:
            ocr_result = response.json()
            print(f"✅ OCR extraction successful!")
            print(f"📊 Processing time: {ocr_result.get('processing_time', 'N/A')}s")
            print(f"📝 Text regions found: {len(ocr_result.get('text_results', []))}")
            
            # Show extracted text
            text_results = ocr_result.get('text_results', [])
            if text_results:
                print(f"\n📝 Extracted text:")
                for i, text in enumerate(text_results[:10], 1):  # Show first 10
                    print(f"  {i}. {text.get('text', 'N/A')} (confidence: {text.get('confidence', 0):.3f})")
                
                if len(text_results) > 10:
                    print(f"  ... and {len(text_results) - 10} more text regions")
            
            print(f"\n✅ PIPELINE RESULT: OCR fallback completed")
            return {
                "success": True,
                "method": "ocr_fallback",
                "text_regions_found": len(text_results),
                "processing_time": ocr_result.get('processing_time', 0),
                "results": text_results
            }
        else:
            print(f"❌ OCR extraction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return {"success": False, "error": "OCR extraction failed"}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ OCR request error: {e}")
        return {"success": False, "error": str(e)}

def check_services():
    """Check if both services are running"""
    print("🔍 Checking services...")
    
    # Check zxing-cpp (barcode detection)
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ zxing-cpp service: Running")
        else:
            print("❌ zxing-cpp service: Not responding")
            return False
    except:
        print("❌ zxing-cpp service: Not running")
        return False
    
    # Check PaddleOCR (OCR fallback)
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ PaddleOCR service: Running")
        else:
            print("❌ PaddleOCR service: Not responding")
            return False
    except:
        print("❌ PaddleOCR service: Not running")
        return False
    
    return True

def main():
    """Main pipeline test"""
    print("🚀 Vision Models Pipeline Test")
    print("=" * 60)
    print("Pipeline: Barcode Detection First → OCR Fallback")
    print()
    
    # Check services
    if not check_services():
        print("\n❌ Services not ready. Please start both containers:")
        print("  - zxing-cpp: http://localhost:8001")
        print("  - PaddleOCR: http://localhost:8000")
        return False
    
    print()
    
    # Test with the specific image
    image_path = Path("../../../test_images/Complete Labels/1_J18CBEP8C7N070812400085N.jpeg")
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Run pipeline test
    start_time = time.time()
    result = test_pipeline_with_image(image_path)
    total_time = time.time() - start_time
    
    print(f"\n📊 PIPELINE SUMMARY")
    print("=" * 60)
    print(f"⏱️  Total pipeline time: {total_time:.2f}s")
    print(f"✅ Success: {result.get('success', False)}")
    print(f"🔧 Method used: {result.get('method', 'unknown')}")
    
    if result.get('success'):
        if result.get('method') == 'bpost_pattern_detection':
            print(f"🎯 BPost pattern found: {result.get('pattern_found', False)}")
            print(f"📋 BPost barcode: {result.get('bpost_barcode', 'N/A')}")
            print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
        elif result.get('method') == 'ocr_fallback':
            print(f"📝 Text regions found: {result.get('text_regions_found', 0)}")
            print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    return result.get('success', False)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
