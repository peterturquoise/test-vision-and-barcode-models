"""
Installation and Setup Script
Downloads required model weights and sets up the environment
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # Install basic requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("Failed to install requirements. Trying individual packages...")
        
        # Install packages individually
        packages = [
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "transformers>=4.30.0",
            "opencv-python>=4.8.0",
            "Pillow>=9.5.0",
            "numpy>=1.24.0",
            "pytesseract>=0.3.10",
            "pyzbar>=0.1.9",
            "ultralytics>=8.0.0",
            "tqdm>=4.65.0"
        ]
        
        for package in packages:
            run_command(f"pip install {package}", f"Installing {package}")

def download_model_weights():
    """Download model weights"""
    print("Downloading model weights...")
    
    # Create models directory
    models_dir = Path("model_weights")
    models_dir.mkdir(exist_ok=True)
    
    # YOLOv9 weights
    print("Downloading YOLOv9 weights...")
    run_command(
        "python -c \"from ultralytics import YOLO; YOLO('yolov9c.pt')\"",
        "Downloading YOLOv9 weights"
    )
    
    # MobileSAM weights
    print("Downloading MobileSAM weights...")
    run_command(
        "python -c \"import requests; requests.get('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_t_4b8939.pth')\"",
        "Downloading MobileSAM weights"
    )

def setup_tesseract():
    """Setup Tesseract OCR"""
    print("Setting up Tesseract OCR...")
    
    # Check if tesseract is installed
    try:
        subprocess.run(["tesseract", "--version"], check=True, capture_output=True)
        print("✓ Tesseract is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Platform-specific installation instructions
    import platform
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        print("Installing Tesseract on macOS...")
        print("Please run: brew install tesseract")
        print("Or download from: https://github.com/tesseract-ocr/tesseract")
        
    elif system == "linux":
        print("Installing Tesseract on Linux...")
        print("Please run: sudo apt-get install tesseract-ocr")
        print("Or: sudo yum install tesseract")
        
    elif system == "windows":
        print("Installing Tesseract on Windows...")
        print("Please download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("And add to PATH")
    
    return False

def create_sample_images():
    """Create sample images for testing"""
    print("Creating sample images...")
    
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    # Create a simple test image with text
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a test image with text
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add some text
        draw.text((50, 50), "Sample Text for Testing", fill='black', font=font)
        draw.text((50, 100), "Vision Models OCR Test", fill='black', font=font)
        
        # Save the image
        img.save(samples_dir / "test_text.png")
        print("✓ Created test_text.png")
        
    except ImportError:
        print("PIL not available, skipping sample image creation")

def main():
    """Main setup function"""
    print("="*80)
    print("VISION MODELS TEST SUITE SETUP")
    print("="*80)
    
    # Install dependencies
    install_dependencies()
    
    # Setup Tesseract
    setup_tesseract()
    
    # Download model weights
    download_model_weights()
    
    # Create sample images
    create_sample_images()
    
    print("\n" + "="*80)
    print("SETUP COMPLETED")
    print("="*80)
    print("Next steps:")
    print("1. Add your test images to the 'samples' folder")
    print("2. Run: python main_test_runner.py")
    print("3. Or run a specific model: python main_test_runner.py --model yolov9")
    print("\nNote: Some models may require additional setup or GPU memory.")
    print("Check individual model files for specific requirements.")

if __name__ == "__main__":
    main()
