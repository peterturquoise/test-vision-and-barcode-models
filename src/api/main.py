#!/usr/bin/env python3
"""
FastAPI Server for Vision Models
Docker-ready API endpoints for model inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import io
from PIL import Image
import json
import time

# Import models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.paddleocr_model import PaddleOCRModel

app = FastAPI(title="Vision Models API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
models = {}

class TextDetectionResponse(BaseModel):
    text: str
    bbox: List[int]
    confidence: float

class BarcodeDetectionResponse(BaseModel):
    bbox: List[int]
    value: str
    barcode_type: str
    confidence: float

class ModelResponse(BaseModel):
    model_name: str
    description: str
    text_detections: List[TextDetectionResponse]
    barcode_detections: List[BarcodeDetectionResponse]
    processing_time: float

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("🚀 Starting Vision Models API...")
    
    # Initialize PaddleOCR
    try:
        print("📦 Loading PaddleOCR...")
        models["paddleocr"] = PaddleOCRModel()
        models["paddleocr"].load_model()
        print("✅ PaddleOCR loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load PaddleOCR: {e}")
        models["paddleocr"] = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vision Models API",
        "version": "1.0.0",
        "available_models": list(models.keys()),
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "analyze": "/analyze/{model_name}",
            "text": "/text/{model_name}",
            "barcodes": "/barcodes/{model_name}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_loaded": {name: model is not None for name, model in models.items()}
    }

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": [
            {
                "name": name,
                "loaded": model is not None,
                "description": model.__class__.__doc__ if model else None
            }
            for name, model in models.items()
        ]
    }

@app.post("/analyze/{model_name}")
async def analyze_image(model_name: str, file: UploadFile = File(...)):
    """Complete image analysis (description + text + barcodes)"""
    if model_name not in models or models[model_name] is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not available")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        model = models[model_name]
        start_time = time.time()
        
        # Get description
        description = model.describe_image(image)
        
        # Extract text
        text_detections = model.extract_text(image)
        
        # Detect barcodes
        barcode_detections = model.detect_barcodes(image)
        
        processing_time = time.time() - start_time
        
        # Format response
        response = ModelResponse(
            model_name=model_name,
            description=description,
            text_detections=[
                TextDetectionResponse(
                    text=det.text,
                    bbox=det.bbox,
                    confidence=det.confidence
                )
                for det in text_detections
            ],
            barcode_detections=[
                BarcodeDetectionResponse(
                    bbox=det.bbox,
                    value=det.value,
                    barcode_type=det.barcode_type,
                    confidence=det.confidence
                )
                for det in barcode_detections
            ],
            processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/text/{model_name}")
async def extract_text(model_name: str, file: UploadFile = File(...)):
    """Extract text only"""
    if model_name not in models or models[model_name] is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not available")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        model = models[model_name]
        start_time = time.time()
        
        text_detections = model.extract_text(image)
        processing_time = time.time() - start_time
        
        return {
            "model_name": model_name,
            "text_detections": [
                {
                    "text": det.text,
                    "bbox": det.bbox,
                    "confidence": det.confidence
                }
                for det in text_detections
            ],
            "processing_time": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

@app.post("/barcodes/{model_name}")
async def detect_barcodes(model_name: str, file: UploadFile = File(...)):
    """Detect barcodes only"""
    if model_name not in models or models[model_name] is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not available")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        model = models[model_name]
        start_time = time.time()
        
        barcode_detections = model.detect_barcodes(image)
        processing_time = time.time() - start_time
        
        return {
            "model_name": model_name,
            "barcode_detections": [
                {
                    "bbox": det.bbox,
                    "value": det.value,
                    "barcode_type": det.barcode_type,
                    "confidence": det.confidence
                }
                for det in barcode_detections
            ],
            "processing_time": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Barcode detection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
