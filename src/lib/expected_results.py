#!/usr/bin/env python3
"""
Shared Expected Results for Package Image Analysis
Loads expected results from .testresults.json files alongside test images
"""

import json
import os
from pathlib import Path

# Default test image (for backward compatibility)
DEFAULT_TEST_IMAGE = "test_images/Complete Labels/3_J18CBEP8CCN070812400095N_90_rot.jpeg"

def analyze_results(text_detections, barcode_detections, model_name="Model", image_path=None):
    """Analyze test results against expected values for any model"""
    print(f"\n📊 {model_name} Results Analysis:")
    print("-" * 40)
    
    # Get expected results for the specific image
    expected_text, expected_barcodes = get_expected_for_image(image_path)
    
    # Check text detections
    found_texts = [det['text'] for det in text_detections]
    text_matches = 0
    
    print("📄 Text Detection Analysis:")
    for expected in expected_text:
        found = any(expected.lower() in text.lower() for text in found_texts)
        status = "✅" if found else "❌"
        print(f"   {status} {expected}")
        if found:
            text_matches += 1
    
    print(f"📊 Text Match Rate: {text_matches}/{len(expected_text)} ({text_matches/len(expected_text)*100:.1f}%)")
    
    # Check barcode detections
    found_barcodes = [det['value'] for det in barcode_detections]
    barcode_matches = 0
    
    print("\n🔍 Barcode Detection Analysis:")
    for expected in expected_barcodes:
        found = any(expected in barcode for barcode in found_barcodes)
        status = "✅" if found else "❌"
        print(f"   {status} {expected}")
        if found:
            barcode_matches += 1
    
    print(f"📊 Barcode Match Rate: {barcode_matches}/{len(expected_barcodes)} ({barcode_matches/len(expected_barcodes)*100:.1f}%)")
    
    # Overall assessment
    total_expected = len(expected_text) + len(expected_barcodes)
    overall_score = (text_matches + barcode_matches) / total_expected * 100 if total_expected > 0 else 0
    print(f"\n🎯 Overall Performance: {overall_score:.1f}%")
    
    if overall_score >= 80:
        print("🌟 Excellent performance!")
    elif overall_score >= 60:
        print("👍 Good performance")
    elif overall_score >= 40:
        print("⚠️ Moderate performance")
    else:
        print("❌ Poor performance - needs improvement")
    
    return {
        'text_matches': text_matches,
        'text_total': len(expected_text),
        'barcode_matches': barcode_matches,
        'barcode_total': len(expected_barcodes),
        'overall_score': overall_score
    }

def load_test_results(image_path):
    """Load expected results from .testresults.json file"""
    if not image_path:
        image_path = DEFAULT_TEST_IMAGE
    
    # Convert to Path object and find the JSON file
    image_path = Path(image_path)
    json_path = image_path.with_suffix(image_path.suffix + '.testresults.json')
    
    if not json_path.exists():
        print(f"⚠️ No .testresults.json file found for {image_path.name}")
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {json_path}: {e}")
        return None

def get_expected_for_image(image_path):
    """Get expected text and barcodes for a specific image from JSON file"""
    results = load_test_results(image_path)
    if not results:
        return [], []
    
    # Extract text and barcode values
    expected_text = results.get('expected_text', [])
    expected_barcodes = []
    
    for barcode_info in results.get('expected_barcodes', []):
        if isinstance(barcode_info, dict):
            expected_barcodes.append(barcode_info.get('value', ''))
        else:
            expected_barcodes.append(str(barcode_info))
    
    return expected_text, expected_barcodes

def get_expected_summary(image_path=None):
    """Get a summary of expected results for a specific image"""
    expected_text, expected_barcodes = get_expected_for_image(image_path)
    return {
        'text_elements': expected_text,
        'barcodes': expected_barcodes,
        'total_elements': len(expected_text) + len(expected_barcodes),
        'test_image': image_path or DEFAULT_TEST_IMAGE
    }

def list_available_images():
    """List all available test images with their .testresults.json files"""
    print("📋 Available Test Images:")
    print("=" * 50)
    
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print("❌ test_images directory not found")
        return
    
    # Scan for .testresults.json files
    json_files = list(test_images_dir.glob("**/*.testresults.json"))
    
    if not json_files:
        print("❌ No .testresults.json files found")
        return
    
    # Group by category
    categories = {}
    for json_file in json_files:
        category = json_file.parent.name
        if category not in categories:
            categories[category] = []
        categories[category].append(json_file)
    
    for category, files in categories.items():
        print(f"\n📁 {category}:")
        for json_file in sorted(files):
            image_file = json_file.with_suffix('').with_suffix('')  # Remove .testresults.json
            image_name = image_file.name
            
            # Load and display info
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                text_count = len(results.get('expected_text', []))
                barcode_count = len(results.get('expected_barcodes', []))
                difficulty = results.get('test_notes', {}).get('difficulty', 'Unknown')
                
                print(f"   📸 {image_name}")
                print(f"      📄 Text: {text_count} elements")
                print(f"      🔍 Barcodes: {barcode_count} elements")
                print(f"      🎯 Difficulty: {difficulty}")
                
            except Exception as e:
                print(f"   📸 {image_name} (❌ Error loading: {e})")

if __name__ == "__main__":
    print("📋 Shared Expected Results for Package Image Analysis")
    print("=" * 60)
    print("Loads expected results from .testresults.json files alongside test images")
    
    # Show available images
    list_available_images()
    
    print(f"\n🎯 Default Test Image: {DEFAULT_TEST_IMAGE}")
    print("\n💡 This benchmark applies to ALL vision models:")
    print("   - PaddleOCR, LLaVA-1.5, YOLOv9, MobileSAM, etc.")
    print("   - Each model can be tested against different image categories")
    print("   - Performance can be compared using this shared benchmark")
    print("\n📝 To add new test images:")
    print("   1. Add the image to appropriate test_images/ subfolder")
    print("   2. Create image_name.jpeg.testresults.json with expected results")
    print("   3. Run tests - results will be automatically loaded")
