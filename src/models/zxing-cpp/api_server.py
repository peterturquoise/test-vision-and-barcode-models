#!/usr/bin/env python3
"""
ZXing-CPP API Server
Fast barcode detection service
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import time
import sys
import re
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from model_correct import ZXingCPPModel

app = FastAPI(title="ZXing-CPP API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = None

@app.on_event("startup")
async def startup_event():
    global model
    print("🚀 Starting ZXing-CPP API server...")
    model = ZXingCPPModel()
    print("✅ ZXing-CPP model loaded successfully!")

@app.get("/")
async def root():
    return {
        "message": "ZXing-CPP API Server",
        "version": "1.0.0",
        "status": "running",
        "model_type": "barcode-only",
        "endpoints": {
            "detect_barcodes": "/detect_barcodes",
            "detect_barcode_pattern": "/detect_barcode_pattern?pattern=3232",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "barcode-only",
        "timestamp": time.time()
    }

@app.post("/detect_barcodes")
async def detect_barcodes(file: UploadFile = File(...)):
    """Detect barcodes in image using correct ZXingReader parsing"""
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Detect barcodes using correct parsing
        start_time = time.time()
        barcode_results = model.detect_barcodes(image)
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "model": "ZXing-CPP",
            "model_type": "barcode-only",
            "barcode_results": barcode_results,
            "processing_time": processing_time,
            "image_size": image.size,
            "image_format": image.format
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting barcodes: {str(e)}")

@app.post("/detect_barcode_pattern")
async def detect_barcode_pattern(file: UploadFile = File(...), pattern: str = Query(..., description="Barcode pattern to search for (e.g., '3232' for BPost)")):
    """
    Fast barcode scanner - stops as soon as it finds a barcode matching the pattern.
    If no matching barcode is found, returns all barcodes found.
    """
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Detect barcodes using correct parsing
        start_time = time.time()
        barcode_results = model.detect_barcodes(image)
        processing_time = time.time() - start_time
        
        # Look for pattern match first
        matching_barcodes = []
        for barcode in barcode_results:
            if barcode["value"].startswith(pattern):
                matching_barcodes.append(barcode)
        
        # Return results
        if matching_barcodes:
            # Found matching pattern - return only the first match (fast scanner mode)
            return {
                "success": True,
                "model": "ZXing-CPP",
                "model_type": "barcode-scanner",
                "pattern": pattern,
                "pattern_found": True,
                "barcode_results": [matching_barcodes[0]],  # Return only first match
                "total_barcodes_found": len(barcode_results),
                "processing_time": processing_time,
                "image_size": image.size,
                "image_format": image.format,
                "message": f"Found barcode matching pattern '{pattern}' - scanner mode"
            }
        else:
            # No pattern match - return all barcodes found
            return {
                "success": True,
                "model": "ZXing-CPP", 
                "model_type": "barcode-scanner",
                "pattern": pattern,
                "pattern_found": False,
                "barcode_results": barcode_results,
                "total_barcodes_found": len(barcode_results),
                "processing_time": processing_time,
                "image_size": image.size,
                "image_format": image.format,
                "message": f"No barcode matching pattern '{pattern}' found - returning all barcodes"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting barcode pattern: {str(e)}")

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    """Describe the image content (not applicable for barcode-only model)"""
    return {
        "message": "ZXing-CPP is a barcode-only detection model. Use /detect_barcodes endpoint.",
        "model_type": "barcode-only"
    }

@app.post("/extract_text")
async def extract_text(file: UploadFile = File(...)):
    """Extract text from image (not applicable for barcode-only model)"""
    return {
        "message": "ZXing-CPP is a barcode-only detection model. Use /detect_barcodes endpoint.",
        "model_type": "barcode-only"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





