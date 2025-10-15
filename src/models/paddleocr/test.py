#!/usr/bin/env python3
"""
PaddleOCR Test
Test PaddleOCR Docker container with expected results analysis
"""

import subprocess
import requests
import time
import sys
from pathlib import Path

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
            "--name", "paddleocr-test",
            "vision-models/paddleocr"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PaddleOCR container started!")
            
            # Wait for container to be ready
            print("⏳ Waiting for container to be ready...")
            time.sleep(30)  # Give it time to load PaddleOCR
            
            return True
        else:
            print(f"❌ Start failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Start error: {e}")
        return False

def test_container_health():
    """Test container health endpoint"""
    print("🏥 Testing container health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_image_analysis():
    """Test image analysis with expected results comparison"""
    print("🔍 Testing image analysis...")
    
    try:
        # Test image path (relative to project root)
        image_path = Path("../../../test_images/Complete Labels/3_J18CBEP8CCN070812400095N_90_rot.jpeg")
        
        if not image_path.exists():
            print(f"❌ Test image not found: {image_path}")
            return False
        
        print(f"📁 Image: {image_path.name}")
        print(f"📂 Path: {image_path}")
        
        # Test both endpoints
        with open(image_path, 'rb') as f:
            files = {'file': f}
            
            # Test describe endpoint
            print("\n🔍 Testing /describe endpoint...")
            response = requests.post("http://localhost:8000/describe", files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Image description successful!")
                print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
                print(f"📝 Description:")
                print(f"   {result['description']}")
            else:
                print(f"❌ Description failed: {response.status_code}")
                print(f"Error: {response.text}")
            
            # Reset file pointer for next request
            f.seek(0)
            
            # Test extract_text endpoint
            print("\n🔍 Testing /extract_text endpoint...")
            response = requests.post("http://localhost:8000/extract_text", files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Text extraction successful!")
                print(f"📄 Text detections: {len(result['text_results'])}")
                print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
                
                # Show ALL text detections
                if result['text_results']:
                    print("\n📄 All extracted text:")
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
                        
                        print(f"   {i+1}. '{text}' (confidence: {confidence:.2f})")
                        print(f"      Bounding box: {bbox}")
                else:
                    print("\n❌ No text detected in the image")
                
                # Analyze results against expected values (text only)
                print("\n" + "="*60)
                analyze_results(result['text_results'], [], "PaddleOCR (Docker)", str(image_path))
                
                return True
            else:
                print(f"❌ Text extraction failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def cleanup_container():
    """Clean up test container"""
    print("🧹 Cleaning up test container...")
    
    try:
        # Stop and remove container
        subprocess.run(["docker", "stop", "paddleocr-test"], capture_output=True)
        subprocess.run(["docker", "rm", "paddleocr-test"], capture_output=True)
        print("✅ Container cleaned up!")
        
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

def main():
    """Main test function"""
    print("🧪 PaddleOCR Docker Container Test")
    print("=" * 50)
    print("Testing PaddleOCR running in Docker container")
    print("with expected results analysis")
    
    try:
        # Start container
        if not start_paddleocr_container():
            return False
        
        # Test health
        if not test_container_health():
            return False
        
        # Test image analysis with expected results
        if not test_image_analysis():
            return False
        
        print("\n✅ All tests passed! PaddleOCR Docker container is working correctly.")
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

