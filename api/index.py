from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="House Price Estimator API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Model from local api directory safely
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "linear_house_price_pipeline.joblib")

# Fallback check if located in parent folder
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "linear_house_price_pipeline.joblib")

artifacts = joblib.load(MODEL_PATH)
pipeline = artifacts["pipeline"]

# 2. Input Schema with Pydantic v1 and v2 compatibility
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

# 3. Serve Frontend directly at root /
@app.get("/", response_class=HTMLResponse)
def serve_home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>House Price Estimator</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 650px; margin: 30px auto; padding: 20px; background: #0f172a; color: #f8fafc; }
        .card { background: #1e293b; padding: 28px; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
        h2 { margin-top: 0; color: #38bdf8; text-align: center; }
        label { display: block; margin-top: 14px; font-weight: 600; font-size: 13px; color: #cbd5e1; }
        input, select { width: 100%; padding: 10px; margin-top: 5px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: #f8fafc; box-sizing: border-box; font-size: 14px; }
        button { width: 100%; margin-top: 24px; padding: 14px; border: none; border-radius: 8px; background: #2563eb; color: white; font-weight: bold; font-size: 16px; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #1d4ed8; }
        #result { margin-top: 20px; padding: 16px; border-radius: 8px; background: #14532d; color: #86efac; text-align: center; display: none; font-size: 18px; font-weight: bold; border: 1px solid #22c55e; }
      </style>
    </head>
    <body>
      <div class="card">
        <h2>🏡 House Price Estimator</h2>
        
        <label>Overall Quality (1: Poor - 10: Excellent)</label>
        <input type="number" id="OverallQual" value="7" min="1" max="10" />

        <label>Above Ground Living Area (sq ft)</label>
        <input type="number" id="GrLivArea" value="1750" />

        <label>Basement Area (sq ft)</label>
        <input type="number" id="TotalBsmtSF" value="950" />

        <label>Garage Car Capacity</label>
        <input type="number" id="GarageCars" value="2" min="0" max="5" />

        <label>Full Bathrooms</label>
        <input type="number" id="FullBath" value="2" min="1" max="4" />

        <label>Year Built</label>
        <input type="number" id="YearBuilt" value="2005" />

        <label>Neighborhood</label>
        <select id="Neighborhood">
          <option value="CollgCr">CollgCr</option>
          <option value="Veenker">Veenker</option>
          <option value="Crawfor">Crawfor</option>
          <option value="NridgHt">NridgHt</option>
          <option value="Mitchel">Mitchel</option>
          <option value="Somerst">Somerst</option>
          <option value="OldTown">OldTown</option>
        </select>

        <label>Exterior Material Quality</label>
        <select id="ExterQual">
          <option value="Ex">Excellent</option>
          <option value="Gd" selected>Good</option>
          <option value="TA">Average</option>
          <option value="Fa">Fair</option>
        </select>

        <label>Kitchen Quality</label>
        <select id="KitchenQual">
          <option value="Ex">Excellent</option>
          <option value="Gd" selected>Good</option>
          <option value="TA">Average</option>
          <option value="Fa">Fair</option>
        </select>

        <button onclick="estimatePrice()">Calculate Estimated Price</button>
        <div id="result"></div>
      </div>

      <script>
        async function estimatePrice() {
          const payload = {
            OverallQual: parseInt(document.getElementById("OverallQual").value),
            GrLivArea: parseFloat(document.getElementById("GrLivArea").value),
            TotalBsmtSF: parseFloat(document.getElementById("TotalBsmtSF").value),
            GarageCars: parseInt(document.getElementById("GarageCars").value),
            FullBath: parseInt(document.getElementById("FullBath").value),
            YearBuilt: parseInt(document.getElementById("YearBuilt").value),
            Neighborhood: document.getElementById("Neighborhood").value,
            ExterQual: document.getElementById("ExterQual").value,
            KitchenQual: document.getElementById("KitchenQual").value,
          };

          const resultBox = document.getElementById("result");
          resultBox.style.display = "block";
          resultBox.style.background = "#1e293b";
          resultBox.style.color = "#94a3b8";
          resultBox.innerText = "Calculating valuation...";

          try {
            const res = await fetch("/api/predict-price", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload)
            });
            const data = await res.json();
            resultBox.style.background = "#14532d";
            resultBox.style.color = "#86efac";
            resultBox.innerText = "Estimated Valuation: $" + Number(data.predicted_price).toLocaleString();
          } catch (err) {
            resultBox.style.background = "#7f1d1d";
            resultBox.style.color = "#fca5a5";
            resultBox.innerText = "Error calculating price.";
          }
        }
      </script>
    </body>
    </html>
    """

# 4. Prediction Endpoint
@app.post("/api/predict-price")
def predict_price(features: HouseFeatures):
    data_dict = features.model_dump() if hasattr(features, "model_dump") else features.dict()
    input_df = pd.DataFrame([data_dict])
    log_pred = pipeline.predict(input_df)[0]
    final_price = float(np.expm1(log_pred))
    return {"predicted_price": round(final_price, 2)}