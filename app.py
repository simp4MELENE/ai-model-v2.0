from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from pathlib import Path
import io
from utils import PREPROCESS_TRANSFORM, load_model

import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Rezbin AI Feature (v2)",
    description="Rezbin's AI feature for classifying images of trash into categories using MobileNetV2.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_dependencies():
    """
    Loads required dependencies for FastAPI on application startup.
    This includes setting up the device (CPU/GPU), model path,
    class labels, and loading the pre-trained MobileNetV2 model.
    """
    app.state.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_file_path = Path(__file__).parent
    app.state.MODEL_PATH = current_file_path / "models" / "MobileNetV2" / "mobilenetv2.pth"
    
    if not app.state.MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {app.state.MODEL_PATH}")

    app.state.CLASS_LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    try:
        app.state.mobilenetv2_model = load_model(
            app.state.MODEL_PATH,
            len(app.state.CLASS_LABELS),
            app.state.DEVICE
        )
        app.state.mobilenetv2_model.eval()
        print(f"Model loaded successfully on {app.state.DEVICE}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

@app.post("/api/predict-trash")
async def predict_trash(file: UploadFile = File(..., description="Image file to classify (PNG, JPG, JPEG).")):
    """
    Receives an image file, preprocesses it, and returns the predicted trash category.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        dict: A dictionary containing the prediction label.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Please upload an image (PNG, JPG, JPEG)."
        )

    try:
        # Read image from uploaded file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess image and move to appropriate device
        img_tensor = PREPROCESS_TRANSFORM(img).unsqueeze(0).to(app.state.DEVICE)
        
        with torch.no_grad():
            outputs = app.state.mobilenetv2_model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            prediction_label = app.state.CLASS_LABELS[predicted_idx.item()]
        
        return {"prediction": prediction_label}
    
    
    
    except Exception as e:
        print(f"Prediction error: {e}") 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during prediction. Please try again."
        )