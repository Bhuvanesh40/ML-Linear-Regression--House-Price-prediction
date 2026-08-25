from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load linear regression artifact
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "linear_house_price_pipeline.joblib")
artifacts = joblib.load(model_path)
pipeline = artifacts["pipeline"]

class HouseFeatures(BaseModel):
    OverallQual: int
    GrLivArea: float
    TotalBsmtSF: float
    GarageCars: int
    FullBath: int
    YearBuilt: int
    Neighborhood: str
    ExterQual: str
    KitchenQual: str

@app.post("/api/predict-price")
def predict_price(features: HouseFeatures):
    input_df = pd.DataFrame([features.dict()])
    log_pred = pipeline.predict(input_df)[0]
    final_price = float(np.expm1(log_pred))
    return {"predicted_price": round(final_price, 2)}