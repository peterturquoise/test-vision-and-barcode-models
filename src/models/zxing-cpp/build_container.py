#!/usr/bin/env python3
"""
ZXing-CPP Container Builder
Build Docker container for ZXing-CPP barcode detection
"""

import subprocess
import sys
import os
from pathlib import Path

def build_container():
    """Build ZXing-CPP Docker container"""
    print("🚀 ZXing-CPP Container Builder")
    print("=" * 40)
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Change to the script directory
    os.chdir(script_dir)
    
    print("🔨 Building ZXing-CPP Docker container...")
    
    try:
        # Build the Docker image
        result = subprocess.run([
            "docker", "build", 
            "-t", "vision-models/zxing-cpp",
            "."
        ], check=True, capture_output=True, text=True)
        
        print("✅ ZXing-CPP container built successfully!")
        print("\n✅ Container build completed successfully!")
        print("You can now run: python test.py")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Container build failed!")
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = build_container()
    if not success:
        sys.exit(1)





