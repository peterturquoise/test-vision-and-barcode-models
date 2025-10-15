#!/usr/bin/env python3
"""
PaddleOCR API Server
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import time
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from model import PaddleOCRModel

app = FastAPI(title="PaddleOCR API", version="1.0.0")

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
    print("🚀 Starting PaddleOCR API server...")
    model = PaddleOCRModel()
    print("✅ PaddleOCR model loaded successfully!")

@app.get("/")
async def root():
    return {
        "message": "PaddleOCR API Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "describe": "/describe",
            "extract_text": "/extract_text", 
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    """Describe the image content"""
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Get description
        start_time = time.time()
        description = model.describe_image(image)
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "description": description,
            "processing_time": processing_time,
            "image_size": image.size,
            "image_format": image.format
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/extract_text")
async def extract_text(file: UploadFile = File(...)):
    """Extract text from image"""
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Extract text
        start_time = time.time()
        text_results = model.extract_text(image)
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "text_results": text_results,
            "processing_time": processing_time,
            "image_size": image.size,
            "image_format": image.format
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

