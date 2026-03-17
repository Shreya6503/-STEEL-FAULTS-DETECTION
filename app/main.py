from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Import prediction helpers from your repo
from predict import predict_batch, load_preprocessing_pipeline
import joblib

app = FastAPI(title="Steel Faults Detection API")

# All 33 input features from your Faults.csv (excluding target columns at the end)
EXPECTED_FEATURES = [
    "X_Min",
    "X_Max",
    "Y_Min",
    "Y_Max",
    "Pix_Area",
    "X_perimeter",
    "Y_perimeter",
    "Sum_of_Luminosity",
    "Minimum_of_Luminosity",
    "Maximum_of_Luminosity",
    "Length_of_Conveyer",
    "TypeOfSteel_A300",
    "TypeOfSteel_A400",
    "Steel_Plate_Thickness",
    "Edges_Index",
    "Empty_Index",
    "Square_Index",
    "Outside_X_Index",
    "Edges_X_Index",
    "Edges_Y_Index",
    "Outside_Global_Index",
    "LogOfAreas",
    "Log_X_Index",
    "Log_Y_Index",
    "Orientation_Index",
    "Luminosity_Index",
    "SigmoidOfAreas",
]


class SingleRecord(BaseModel):
    """Single row of features matching the training data schema."""

    X_Min: float
    X_Max: float
    Y_Min: float
    Y_Max: float
    Pix_Area: float
    X_perimeter: float
    Y_perimeter: float
    Sum_of_Luminosity: float
    Minimum_of_Luminosity: float
    Maximum_of_Luminosity: float
    Length_of_Conveyer: float
    TypeOfSteel_A300: float
    TypeOfSteel_A400: float
    Steel_Plate_Thickness: float
    Edges_Index: float
    Empty_Index: float
    Square_Index: float
    Outside_X_Index: float
    Edges_X_Index: float
    Edges_Y_Index: float
    Outside_Global_Index: float
    LogOfAreas: float
    Log_X_Index: float
    Log_Y_Index: float
    Orientation_Index: float
    Luminosity_Index: float
    SigmoidOfAreas: float


class BatchPayload(BaseModel):
    """Batch prediction request: list of feature records."""

    data: List[SingleRecord]


# Global model & preprocessors (loaded at startup)
MODEL = None
SCALER = None
FEATURE_SELECTOR = None
LABEL_ENCODER = None


@app.on_event("startup")
def startup_load_model():
    global MODEL, SCALER, FEATURE_SELECTOR, LABEL_ENCODER
    MODEL = None
    SCALER = None
    FEATURE_SELECTOR = None
    LABEL_ENCODER = None

    try:
        pipeline = load_preprocessing_pipeline(
            "models/scaler.pkl",
            "models/feature_selector.pkl",
            "models/label_encoder.pkl",
        )
        SCALER = pipeline.get("scaler")
        FEATURE_SELECTOR = pipeline.get("feature_selector")
        LABEL_ENCODER = pipeline.get("label_encoder")
    except Exception as e:
        print(f"Warning: Could not load preprocessing pipeline: {e}")

    try:
        MODEL = joblib.load("models/steel_fault_model.pkl")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "preprocessors_loaded": SCALER is not None and FEATURE_SELECTOR is not None,
    }


@app.post("/predict")
def predict_endpoint(payload: BatchPayload, return_labels: Optional[bool] = True):
    """
    POST /predict - Make batch predictions on steel fault data.
    
    Request body: JSON with list of records, each with 27 numeric features.
    Returns: predictions and probabilities for each record.
    """
    if MODEL is None or SCALER is None or FEATURE_SELECTOR is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessors not loaded. Check models/ folder for: model.pkl, scaler.pkl, feature_selector.pkl, label_encoder.pkl",
        )

    try:
        # Convert list of Pydantic models to list of dicts
        records = [record.dict() for record in payload.data]
        
        # Create DataFrame explicitly with feature columns only
        df = pd.DataFrame(records)
        
        # Get feature names from the scaler (what it was trained on)
        scaler_features = SCALER.get_feature_names_out() if hasattr(SCALER, 'get_feature_names_out') else EXPECTED_FEATURES
        
        # Verify all required features are present
        missing = set(scaler_features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Select only features in correct order (as scaler expects)
        df = df[scaler_features]

        # Use predict_batch from predict.py
        result = predict_batch(
            MODEL, df, SCALER, FEATURE_SELECTOR, LABEL_ENCODER if return_labels else None
        )

        preds = result.get("predictions")
        probs = result.get("probabilities")

        # Convert numpy arrays to JSON-serializable lists
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
        if isinstance(probs, np.ndarray):
            probs = probs.tolist()

        return {"predictions": preds, "probabilities": probs}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required feature: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
