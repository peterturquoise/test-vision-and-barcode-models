#!/usr/bin/env python3
"""
Test script to analyze corrupted barcodes and extract partial information
for matching against preadvice files.
"""

import requests
import json
import re
from typing import List, Dict, Tuple

def analyze_corrupted_barcode(barcode_value: str, error_type: str) -> Dict:
    """
    Analyze a corrupted barcode to extract partial information.
    
    Args:
        barcode_value: The corrupted barcode text
        error_type: The type of error (e.g., "ChecksumError")
    
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "original": barcode_value,
        "error_type": error_type,
        "partial_matches": [],
        "confidence": "low"
    }
    
    # For BPost barcodes (starting with 3232)
    if barcode_value.startswith("3232"):
        analysis["barcode_type"] = "BPost"
        analysis["confidence"] = "high"  # Even corrupted, we know it's BPost
        
        # Extract partial information
        if len(barcode_value) >= 6:
            analysis["partial_matches"].append({
                "type": "prefix",
                "value": barcode_value[:6],  # "3232XX"
                "description": "BPost prefix with partial ID"
            })
        
        if len(barcode_value) >= 10:
            analysis["partial_matches"].append({
                "type": "prefix_extended", 
                "value": barcode_value[:10],  # "3232XXXXXX"
                "description": "BPost prefix with extended partial ID"
            })
    
    # For other barcodes, try to extract meaningful parts
    else:
        # Look for common patterns
        if re.match(r'^\d{10,}$', barcode_value):
            analysis["barcode_type"] = "Numeric"
            analysis["partial_matches"].append({
                "type": "numeric_prefix",
                "value": barcode_value[:8],
                "description": "First 8 digits"
            })
        
        elif re.match(r'^[A-Z0-9]{8,}$', barcode_value):
            analysis["barcode_type"] = "Alphanumeric"
            analysis["partial_matches"].append({
                "type": "alphanumeric_prefix",
                "value": barcode_value[:6],
                "description": "First 6 characters"
            })
        else:
            analysis["barcode_type"] = "Unknown"
            analysis["partial_matches"].append({
                "type": "generic_prefix",
                "value": barcode_value[:8],
                "description": "First 8 characters"
            })
    
    return analysis

def test_corruption_analysis():
    """Test the corruption analysis with the actual API results."""
    
    print("🔍 Testing Corruption Analysis for Barcode Matching")
    print("=" * 60)
    
    # Test with the actual image
    image_path = "/Users/peterbrooke/dev/cursor/test-7-vision-models/test_images/Complete Labels/1_J18CBEP8C7N070812400085N.jpeg"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:8001/detect_barcodes', files=files)
        
        if response.status_code == 200:
            data = response.json()
            barcodes = data.get('barcode_results', [])
            
            print(f"📊 Found {len(barcodes)} barcodes")
            print()
            
            for i, barcode in enumerate(barcodes, 1):
                print(f"🔍 Barcode {i}:")
                print(f"   Value: {barcode['value']}")
                print(f"   Type: {barcode['barcode_type']}")
                print(f"   Confidence: {barcode.get('confidence', 'N/A')}%")
                print(f"   Has Error: {barcode.get('has_error', False)}")
                print(f"   Error Type: {barcode.get('error_type', 'None')}")
                
                # Analyze corrupted barcodes
                if barcode.get('has_error', False):
                    analysis = analyze_corrupted_barcode(
                        barcode['value'], 
                        barcode.get('error_type', 'Unknown')
                    )
                    
                    print(f"   📋 Corruption Analysis:")
                    print(f"      Barcode Type: {analysis['barcode_type']}")
                    print(f"      Confidence: {analysis['confidence']}")
                    print(f"      Partial Matches:")
                    
                    for match in analysis['partial_matches']:
                        print(f"        - {match['type']}: {match['value']} ({match['description']})")
                    
                    # Show how this could be used for preadvice matching
                    if analysis['barcode_type'] == 'BPost':
                        print(f"   🎯 Preadvice Matching Strategy:")
                        print(f"      - Search for barcodes starting with: {barcode['value'][:6]}")
                        print(f"      - Use fuzzy matching for: {barcode['value'][:10]}")
                        print(f"      - Priority: HIGH (BPost barcode detected)")
                
                print()
        
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Error: {e}")

def demonstrate_preadvice_matching():
    """Demonstrate how corrupted barcodes could be used for preadvice matching."""
    
    print("🎯 Preadvice Matching Strategy Demonstration")
    print("=" * 50)
    
    # Simulate corrupted barcodes from different scenarios
    test_cases = [
        {
            "barcode": "323201017200000687647030",  # Corrupted BPost
            "error": "ChecksumError",
            "clean_version": "323201167200000687647030"
        },
        {
            "barcode": "4304P91483",  # Corrupted Code128
            "error": "ChecksumError", 
            "clean_version": "4304591483"
        },
        {
            "barcode": "1234567890",  # Partial numeric
            "error": "PartialRead",
            "clean_version": "1234567890123456"
        }
    ]
    
    for case in test_cases:
        print(f"\n📋 Test Case: {case['barcode']}")
        analysis = analyze_corrupted_barcode(case['barcode'], case['error'])
        
        print(f"   Error: {case['error']}")
        print(f"   Clean Version: {case['clean_version']}")
        print(f"   Analysis: {analysis['barcode_type']} ({analysis['confidence']} confidence)")
        
        print(f"   🔍 Matching Strategies:")
        for match in analysis['partial_matches']:
            print(f"      - {match['type']}: '{match['value']}' → Search preadvice for barcodes starting with this")
        
        # Show SQL-like query for preadvice matching
        if analysis['barcode_type'] == 'BPost':
            print(f"   💾 Preadvice Query:")
            print(f"      SELECT * FROM preadvice WHERE barcode LIKE '{case['barcode'][:6]}%'")
            print(f"      ORDER BY similarity_score DESC LIMIT 10")

if __name__ == "__main__":
    test_corruption_analysis()
    print("\n" + "="*60 + "\n")
    demonstrate_preadvice_matching()
