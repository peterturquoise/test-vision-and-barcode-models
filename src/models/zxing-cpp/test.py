#!/usr/bin/env python3
"""
ZXing-CPP Test
Test ZXing-CPP Docker container with expected results analysis
"""

import subprocess
import requests
import time
import sys
from pathlib import Path

# Add paths for expected results analysis
sys.path.append(str(Path(__file__).parent.parent.parent / "lib"))
from expected_results import analyze_results

def start_zxing_container():
    """Start ZXing-CPP container"""
    print("🚀 Starting ZXing-CPP container...")
    
    try:
        # Start the container
        result = subprocess.run([
            "docker", "run", 
            "--name", "zxing-test",
            "-p", "8001:8000",  # Use different port to avoid conflicts
            "-d",
            "vision-models/zxing-cpp"
        ], check=True, capture_output=True, text=True)
        
        print("✅ ZXing-CPP container started!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Start failed: {e.stderr}")
        return False

def wait_for_container():
    """Wait for container to be ready"""
    print("⏳ Waiting for container to be ready...")
    
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                print("✅ Container is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
    
    print("❌ Container failed to start within 30 seconds")
    return False

def test_container_health():
    """Test container health"""
    print("🏥 Testing container health...")
    
    try:
        response = requests.get("http://localhost:8001/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_barcode_detection():
    """Test barcode detection on test image"""
    print("🔍 Testing barcode detection...")
    
    # Path to test image
    test_image_path = Path(__file__).parent.parent.parent.parent / "test_images" / "Complete Labels" / "3_J18CBEP8CCN070812400095N_90_rot.jpeg"
    
    if not test_image_path.exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    try:
        # Send image to container
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8001/detect_barcodes", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Barcode detection successful!")
            print(f"Processing time: {result.get('processing_time', 'N/A')}s")
            print(f"Barcodes found: {len(result.get('barcode_results', []))}")
            
            # Show detected barcodes
            for i, barcode in enumerate(result.get('barcode_results', [])):
                print(f"  Barcode {i+1}: {barcode.get('value', 'N/A')} ({barcode.get('barcode_type', 'N/A')})")
            
            # Analyze results against expected data
            print("\n📊 Analyzing results against expected data...")
            try:
                # Convert results to expected format
                barcode_detections = []
                for barcode in result.get('barcode_results', []):
                    # Create a simple barcode detection object
                    class SimpleBarcodeDetection:
                        def __init__(self, bbox, value, barcode_type, confidence):
                            self.bbox = bbox
                            self.value = value
                            self.barcode_type = barcode_type
                            self.confidence = confidence
                    
                    barcode_detections.append(SimpleBarcodeDetection(
                        bbox=barcode.get('bbox', [0, 0, 0, 0]),
                        value=barcode.get('value', ''),
                        barcode_type=barcode.get('barcode_type', 'UNKNOWN'),
                        confidence=barcode.get('confidence', 0.0)
                    ))
                
                # Analyze results
                analysis = analyze_results([], barcode_detections, "ZXing-CPP", str(test_image_path))
                print(f"Analysis: {analysis}")
                
            except Exception as e:
                print(f"⚠️ Could not analyze results: {e}")
            
            return True
        else:
            print(f"❌ Barcode detection failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False

def cleanup_container():
    """Clean up test container"""
    print("🧹 Cleaning up test container...")
    
    try:
        # Stop and remove container
        subprocess.run(["docker", "stop", "zxing-test"], capture_output=True)
        subprocess.run(["docker", "rm", "zxing-test"], capture_output=True)
        print("✅ Container cleaned up!")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

def main():
    """Main test function"""
    print("🧪 ZXing-CPP Docker Container Test")
    print("=" * 50)
    print("Testing ZXing-CPP running in Docker container")
    print("with expected results analysis")
    
    try:
        # Start container
        if not start_zxing_container():
            return False
        
        # Wait for container to be ready
        if not wait_for_container():
            cleanup_container()
            return False
        
        # Test health
        if not test_container_health():
            cleanup_container()
            return False
        
        # Test barcode detection
        if not test_barcode_detection():
            cleanup_container()
            return False
        
        print("\n✅ All tests passed!")
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
    finally:
        cleanup_container()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)





