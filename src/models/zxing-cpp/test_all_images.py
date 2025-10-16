#!/usr/bin/env python3
"""
Comprehensive ZXing-CPP API Test
Test the fast raw barcode API with all images in test_images directory
"""

import requests
import time
import json
from pathlib import Path
import sys

def test_all_images():
    """Test the fast raw barcode API with all images"""
    print("🚀 Comprehensive ZXing-CPP API Test")
    print("=" * 60)
    print("Testing fast raw barcode detection on all test images")
    print()
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code != 200:
            print("❌ ZXing-CPP API not running. Please start the container.")
            return False
        print("✅ ZXing-CPP API is running")
    except:
        print("❌ ZXing-CPP API not accessible. Please start the container.")
        return False
    
    # Get all test images
    test_images_dir = Path("../../../test_images")
    if not test_images_dir.exists():
        print(f"❌ Test images directory not found: {test_images_dir}")
        return False
    
    # Find all image files
    image_files = []
    for subdir in test_images_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            for file in subdir.iterdir():
                if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_files.append(file)
    
    if not image_files:
        print("❌ No image files found in test_images directory")
        return False
    
    print(f"📁 Found {len(image_files)} images to test")
    print()
    
    # Test results
    results = []
    total_time = 0
    total_barcodes = 0
    
    # Test each image
    for i, image_path in enumerate(image_files, 1):
        print(f"🔍 Testing {i}/{len(image_files)}: {image_path.name}")
        print(f"   📁 Directory: {image_path.parent.name}")
        
        try:
            # Test /detect_barcodes endpoint
            start_time = time.time()
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post("http://localhost:8001/detect_barcodes", files=files, timeout=30)
            
            processing_time = time.time() - start_time
            total_time += processing_time
            
            if response.status_code == 200:
                result = response.json()
                barcodes = result.get('barcode_results', [])
                total_barcodes += len(barcodes)
                
                print(f"   ✅ Success: {len(barcodes)} barcodes in {processing_time:.3f}s")
                
                # Show barcodes found
                for j, barcode in enumerate(barcodes, 1):
                    error_info = f" (ERROR: {barcode.get('error_type', 'None')})" if barcode.get('has_error') else ""
                    print(f"      {j}. {barcode['value']} ({barcode['barcode_type']}){error_info}")
                
                # Test pattern detection for BPost images
                if 'Complete Labels' in str(image_path) or 'BPost' in str(image_path):
                    print(f"   🎯 Testing BPost pattern detection...")
                    pattern_start = time.time()
                    with open(image_path, 'rb') as f:
                        files = {'file': f}
                        pattern_response = requests.post("http://localhost:8001/detect_barcode_pattern?pattern=3232", files=files, timeout=30)
                    
                    pattern_time = time.time() - pattern_start
                    
                    if pattern_response.status_code == 200:
                        pattern_result = pattern_response.json()
                        pattern_found = pattern_result.get('pattern_found', False)
                        print(f"      Pattern '3232' found: {pattern_found} in {pattern_time:.3f}s")
                        
                        if pattern_found:
                            pattern_barcodes = pattern_result.get('barcode_results', [])
                            for barcode in pattern_barcodes:
                                print(f"      🎯 BPost: {barcode['value']} ({barcode['barcode_type']})")
                
                results.append({
                    'image': image_path.name,
                    'directory': image_path.parent.name,
                    'success': True,
                    'barcodes_found': len(barcodes),
                    'processing_time': processing_time,
                    'barcodes': barcodes
                })
                
            else:
                print(f"   ❌ Failed: HTTP {response.status_code}")
                results.append({
                    'image': image_path.name,
                    'directory': image_path.parent.name,
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'barcodes_found': 0,
                    'processing_time': processing_time
                })
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request error: {e}")
            results.append({
                'image': image_path.name,
                'directory': image_path.parent.name,
                'success': False,
                'error': str(e),
                'barcodes_found': 0,
                'processing_time': 0
            })
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'image': image_path.name,
                'directory': image_path.parent.name,
                'success': False,
                'error': str(e),
                'barcodes_found': 0,
                'processing_time': 0
            })
        
        print()
    
    # Summary
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    print(f"📁 Total images tested: {len(image_files)}")
    print(f"✅ Successful tests: {sum(1 for r in results if r['success'])}")
    print(f"❌ Failed tests: {sum(1 for r in results if not r['success'])}")
    print(f"📋 Total barcodes found: {total_barcodes}")
    print(f"⏱️  Total processing time: {total_time:.2f}s")
    print(f"📊 Average time per image: {total_time/len(image_files):.3f}s")
    print(f"📊 Average barcodes per image: {total_barcodes/len(image_files):.1f}")
    
    # Results by directory
    print(f"\n📁 RESULTS BY DIRECTORY:")
    print("-" * 40)
    directories = {}
    for result in results:
        dir_name = result['directory']
        if dir_name not in directories:
            directories[dir_name] = {'total': 0, 'success': 0, 'barcodes': 0, 'time': 0}
        directories[dir_name]['total'] += 1
        if result['success']:
            directories[dir_name]['success'] += 1
            directories[dir_name]['barcodes'] += result['barcodes_found']
            directories[dir_name]['time'] += result['processing_time']
    
    for dir_name, stats in directories.items():
        success_rate = (stats['success'] / stats['total']) * 100
        avg_time = stats['time'] / stats['success'] if stats['success'] > 0 else 0
        avg_barcodes = stats['barcodes'] / stats['success'] if stats['success'] > 0 else 0
        print(f"  {dir_name}: {stats['success']}/{stats['total']} ({success_rate:.1f}%) - {avg_barcodes:.1f} barcodes/image, {avg_time:.3f}s/image")
    
    # Save detailed results
    results_file = Path("comprehensive_test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total_images': len(image_files),
                'successful_tests': sum(1 for r in results if r['success']),
                'failed_tests': sum(1 for r in results if not r['success']),
                'total_barcodes': total_barcodes,
                'total_time': total_time,
                'average_time_per_image': total_time/len(image_files),
                'average_barcodes_per_image': total_barcodes/len(image_files)
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return True

def main():
    """Main test function"""
    try:
        success = test_all_images()
        return success
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
