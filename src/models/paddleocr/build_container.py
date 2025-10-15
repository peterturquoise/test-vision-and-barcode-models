#!/usr/bin/env python3
"""
PaddleOCR Container Builder
Build PaddleOCR Docker container
"""

import subprocess
from pathlib import Path

def build_paddleocr_container():
    """Build PaddleOCR Docker container"""
    print("🔨 Building PaddleOCR Docker container...")
    
    try:
        # Build the container from current directory
        result = subprocess.run([
            "docker", "build", "-t", "vision-models/paddleocr", "."
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PaddleOCR container built successfully!")
            return True
        else:
            print(f"❌ Build failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False

def main():
    """Main build function"""
    print("🚀 PaddleOCR Container Builder")
    print("=" * 40)
    
    if build_paddleocr_container():
        print("\n✅ Container build completed successfully!")
        print("You can now run: python test_container.py")
    else:
        print("\n❌ Container build failed!")

if __name__ == "__main__":
    main()






