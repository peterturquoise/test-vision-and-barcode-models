#!/usr/bin/env python3
"""
Comprehensive PaddleOCR Test for All Images
Test PaddleOCR Docker container with all test images and save results
"""

import subprocess
import requests
import time
import sys
import json
from pathlib import Path
from datetime import datetime

# Add paths for expected results analysis
sys.path.append(str(Path(__file__).parent.parent.parent / "lib"))
from expected_results import analyze_results

def start_paddleocr_container():
    """Start PaddleOCR container"""
    print("🚀 Starting PaddleOCR container...")
    
    try:
        # Start container in background
        result = subprocess.run([
            "docker", "run", "-d", 
            "-p", "8000:8000",
            "--name", "paddleocr-test-all",
            "vision-models/paddleocr"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PaddleOCR container started!")
            
            # Wait for container to be ready
            print("⏳ Waiting for container to be ready...")
            time.sleep(45)  # Give it more time to load PaddleOCR
            
            return True
        else:
            print(f"❌ Start failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Start error: {e}")
        return False

def test_container_health():
    """Test container health endpoint with retries"""
    print("🏥 Testing container health...")
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed: {health_data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code} (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"❌ Health check error (attempt {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            print("⏳ Retrying in 10 seconds...")
            time.sleep(10)
    
    print("❌ Health check failed after all retries")
    return False

def process_image(image_path, category):
    """Process a single image and return results"""
    print(f"\n🔍 Processing: {image_path.name} ({category})")
    
    try:
        # Test both endpoints
        with open(image_path, 'rb') as f:
            files = {'file': f}
            
            # Test describe endpoint
            response = requests.post("http://localhost:8000/describe", files=files, timeout=60)
            
            description_result = None
            if response.status_code == 200:
                description_result = response.json()
                print(f"✅ Description: {description_result['description'][:100]}...")
            else:
                print(f"❌ Description failed: {response.status_code}")
            
            # Reset file pointer for next request
            f.seek(0)
            
            # Test extract_text endpoint
            response = requests.post("http://localhost:8000/extract_text", files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Text extraction successful!")
                print(f"📄 Text detections: {len(result['text_results'])}")
                print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
                
                # Show ALL text detections
                if result['text_results']:
                    print(f"📄 Extracted text:")
                    for i, detection in enumerate(result['text_results']):
                        # Handle both dict and object formats
                        if isinstance(detection, dict):
                            text = detection.get('text', 'N/A')
                            confidence = detection.get('confidence', 0.0)
                            bbox = detection.get('bbox', [])
                        else:
                            text = getattr(detection, 'text', 'N/A')
                            confidence = getattr(detection, 'confidence', 0.0)
                            bbox = getattr(detection, 'bbox', [])
                        
                        print(f"   {i+1:2d}. '{text}' (conf: {confidence:.3f})")
                else:
                    print("❌ No text detected in the image")
                
                # Analyze results against expected values
                print("\n" + "="*60)
                analyze_results(result['text_results'], [], "PaddleOCR (Docker)", str(image_path))
                
                return {
                    'success': True,
                    'image_name': image_path.name,
                    'category': category,
                    'description': description_result['description'] if description_result else None,
                    'text_results': result['text_results'],
                    'processing_time': result['processing_time'],
                    'total_detections': len(result['text_results']),
                    'image_size': result.get('image_size', [0, 0]),
                    'image_format': result.get('image_format', 'Unknown')
                }
            else:
                print(f"❌ Text extraction failed: {response.status_code}")
                print(f"Error: {response.text}")
                return {
                    'success': False,
                    'image_name': image_path.name,
                    'category': category,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
            
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return {
            'success': False,
            'image_name': image_path.name,
            'category': category,
            'error': str(e)
        }

def cleanup_container():
    """Clean up test container"""
    print("\n🧹 Cleaning up test container...")
    
    try:
        # Stop and remove container
        subprocess.run(["docker", "stop", "paddleocr-test-all"], capture_output=True)
        subprocess.run(["docker", "rm", "paddleocr-test-all"], capture_output=True)
        print("✅ Container cleaned up!")
        
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

def main():
    """Main test function"""
    print("🧪 PaddleOCR Docker Container - All Images Test")
    print("=" * 60)
    print("Testing PaddleOCR with all images in test_images directory")
    
    # Define test image directories
    test_images_dir = Path("../../../test_images")
    categories = {
        "Complete Labels": test_images_dir / "Complete Labels",
        "Broken Barcode": test_images_dir / "Broken Barcode", 
        "Fragment Labels": test_images_dir / "Fragment Labels"
    }
    
    # Collect all images
    all_images = []
    for category, category_dir in categories.items():
        if category_dir.exists():
            for image_file in category_dir.glob("*.jpeg"):
                if not image_file.name.endswith('.testresults.json'):
                    all_images.append((image_file, category))
    
    print(f"📁 Found {len(all_images)} images to process:")
    for image_path, category in all_images:
        print(f"   - {category}: {image_path.name}")
    
    try:
        # Start container
        if not start_paddleocr_container():
            return False
        
        # Test health
        if not test_container_health():
            return False
        
        # Process all images
        results = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'total_images': len(all_images),
                'categories': list(categories.keys()),
                'paddleocr_version': 'Docker Container',
                'test_type': 'All Images Comprehensive Test'
            },
            'results': []
        }
        
        successful_tests = 0
        failed_tests = 0
        
        for i, (image_path, category) in enumerate(all_images, 1):
            print(f"\n{'='*80}")
            print(f"📸 Processing Image {i}/{len(all_images)}")
            print(f"{'='*80}")
            
            result = process_image(image_path, category)
            results['results'].append(result)
            
            if result['success']:
                successful_tests += 1
            else:
                failed_tests += 1
        
        # Save results to JSON file
        results_file = Path("../../../results/PaddleOCR_all_images_results.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print("📊 FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Successful tests: {successful_tests}")
        print(f"❌ Failed tests: {failed_tests}")
        print(f"📁 Total images processed: {len(all_images)}")
        print(f"💾 Results saved to: {results_file}")
        
        # Category breakdown
        category_stats = {}
        for result in results['results']:
            category = result['category']
            if category not in category_stats:
                category_stats[category] = {'success': 0, 'failed': 0, 'total_detections': 0}
            
            if result['success']:
                category_stats[category]['success'] += 1
                category_stats[category]['total_detections'] += result.get('total_detections', 0)
            else:
                category_stats[category]['failed'] += 1
        
        print(f"\n📊 Category Breakdown:")
        for category, stats in category_stats.items():
            total = stats['success'] + stats['failed']
            avg_detections = stats['total_detections'] / stats['success'] if stats['success'] > 0 else 0
            print(f"   {category}: {stats['success']}/{total} successful, avg {avg_detections:.1f} detections")
        
        print(f"\n✅ All tests completed! PaddleOCR Docker container processed all images.")
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    finally:
        cleanup_container()

if __name__ == "__main__":
    main()
