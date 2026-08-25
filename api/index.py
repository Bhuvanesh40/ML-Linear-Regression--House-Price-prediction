from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import time

app = FastAPI(title="Machine Learning Master Suite API")

APP_START_TIME = time.time()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global caches for model artifacts
MODEL_CACHE = {}

def get_artifact(name: str):
    if name not in MODEL_CACHE:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        paths_to_try = [
            os.path.join(curr_dir, name),
            os.path.join(os.path.dirname(curr_dir), name)
        ]
        for p in paths_to_try:
            if os.path.exists(p):
                MODEL_CACHE[name] = joblib.load(p)
                break
        if name not in MODEL_CACHE:
            raise RuntimeError(f"Model artifact file '{name}' not found.")
    return MODEL_CACHE[name]


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================
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


class ChurnFeatures(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str
    OnlineSecurity: str
    TechSupport: str
    Algorithm: str = "Random Forest"


# ==============================================================================
# HTML & JS FRONTEND DASHBOARD (STREAMLIT NATIVE DARK THEME AESTHETIC)
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Machine Learning Master Suite</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg-main: #0e1117;
      --bg-sidebar: #262730;
      --bg-card: #1e222b;
      --border-color: #31333f;
      --text-main: #fafafa;
      --text-muted: #a3a8b8;
      --primary-red: #ff4b4b;
      --primary-red-hover: #ff2b2b;
      --success-green: #21c354;
      --success-bg: #0d381e;
      --error-bg: #4a151b;
      --error-red: #ff6b6b;
      --accent-blue: #58a6ff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg-main);
      color: var(--text-main);
      display: flex;
      min-height: 100vh;
    }
    
    /* SIDEBAR */
    #sidebar {
      width: 290px;
      min-width: 290px;
      background: var(--bg-sidebar);
      border-right: 1px solid var(--border-color);
      padding: 24px 18px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    #sidebar h2 {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .sidebar-label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .radio-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .radio-option {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.88rem;
      color: var(--text-muted);
      transition: all 0.15s ease;
      user-select: none;
    }
    .radio-option:hover {
      background: rgba(255, 255, 255, 0.05);
      color: var(--text-main);
    }
    .radio-option.active {
      background: rgba(255, 75, 75, 0.15);
      color: var(--text-main);
      font-weight: 600;
    }
    .radio-dot {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      border: 2px solid #6b7280;
      margin-top: 2px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.15s ease;
    }
    .radio-option.active .radio-dot {
      border-color: var(--primary-red);
      background: var(--primary-red);
      box-shadow: 0 0 6px rgba(255, 75, 75, 0.6);
    }

    /* MAIN CONTENT */
    #main-content {
      flex: 1;
      padding: 36px 48px;
      overflow-y: auto;
      max-width: 1100px;
    }
    .pillar-pane { display: none; }
    .pillar-pane.active { display: block; animation: fadeIn 0.2s ease; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

    h1.page-title {
      font-size: 1.85rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    p.page-desc {
      color: var(--text-muted);
      font-size: 0.92rem;
      margin-bottom: 20px;
    }
    .divider {
      height: 1px;
      background: var(--border-color);
      margin: 20px 0 24px 0;
    }

    /* METRICS ROW */
    .metrics-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .metric-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      padding: 16px 20px;
      border-radius: 8px;
    }
    .metric-card span {
      display: block;
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    .metric-card strong {
      font-size: 1.4rem;
      font-weight: 700;
      color: var(--text-main);
    }

    /* FORM GRID */
    h3.section-header {
      font-size: 1.15rem;
      font-weight: 600;
      color: var(--text-main);
      margin: 20px 0 12px 0;
    }
    .grid-3 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 20px;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
    }
    .form-group {
      margin-bottom: 14px;
    }
    label {
      display: block;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    input[type="number"], select {
      width: 100%;
      padding: 10px 12px;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      color: var(--text-main);
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.15s ease;
    }
    input[type="number"]:focus, select:focus {
      border-color: var(--primary-red);
    }
    .slider-container {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    input[type="range"] {
      flex: 1;
      accent-color: var(--primary-red);
    }
    .slider-val {
      font-weight: 700;
      color: var(--primary-red);
      min-width: 24px;
      text-align: right;
    }

    /* BUTTONS */
    .st-btn {
      width: 100%;
      background: var(--primary-red);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 12px 20px;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      margin-top: 20px;
      transition: background 0.15s ease, transform 0.05s;
    }
    .st-btn:hover {
      background: var(--primary-red-hover);
    }
    .st-btn:active {
      transform: scale(0.99);
    }

    /* RESULT ALERTS */
    .st-alert {
      margin-top: 20px;
      padding: 16px 20px;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      display: none;
    }
    .st-alert.success {
      background: var(--success-bg);
      border: 1px solid #1c6b34;
      color: #85e89d;
    }
    .st-alert.error {
      background: var(--error-bg);
      border: 1px solid #8e2029;
      color: #ff9b9b;
    }
    .st-alert.info {
      background: #102a43;
      border: 1px solid #205493;
      color: #90cdf4;
      display: block;
    }

    /* TABLE */
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      background: var(--bg-card);
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border-color);
      font-size: 0.88rem;
    }
    th, td {
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border-color);
    }
    th {
      background: #14171f;
      color: var(--text-muted);
      font-weight: 600;
    }
    tr:hover { background: rgba(255, 255, 255, 0.02); }

    /* CHARTS */
    .chart-box {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
      margin-top: 14px;
      position: relative;
      height: 420px;
    }
  </style>
</head>
<body>

  <!-- SIDEBAR -->
  <div id="sidebar">
    <h2>🤖 ML Algorithms Suite</h2>
    <div>
      <div class="sidebar-label">Select Machine Learning Pillar:</div>
      <div class="radio-group">
        <div class="radio-option active" onclick="switchPillar('pillar1', this)">
          <div class="radio-dot"></div>
          <span>1. Regression (Linear / Ridge)</span>
        </div>
        <div class="radio-option" onclick="switchPillar('pillar2', this)">
          <div class="radio-dot"></div>
          <span>2. Classification Suite (5 Models)</span>
        </div>
        <div class="radio-option" onclick="switchPillar('pillar3', this)">
          <div class="radio-dot"></div>
          <span>3. Clustering Suite (3 Models)</span>
        </div>
        <div class="radio-option" onclick="switchPillar('pillar4', this)">
          <div class="radio-dot"></div>
          <span>4. Dimensionality Reduction (PCA &amp; t-SNE)</span>
        </div>
      </div>
    </div>
  </div>

  <!-- MAIN VIEW -->
  <div id="main-content">

    <!-- PILLAR 1: REGRESSION -->
    <div id="pillar1" class="pillar-pane active">
      <h1 class="page-title">🏡 House Price Estimator (Linear Regression)</h1>
      <p class="page-desc">Predict property sales prices using <strong>Regularized Linear Regression (Ridge)</strong> on the Ames Housing dataset.</p>
      <div class="divider"></div>

      <div class="metrics-row">
        <div class="metric-card">
          <span>Validation R² Score</span>
          <strong>0.8720</strong>
        </div>
        <div class="metric-card">
          <span>Validation RMSLE</span>
          <strong>0.1546</strong>
        </div>
        <div class="metric-card">
          <span>Validation MAE</span>
          <strong>$18,635.03</strong>
        </div>
      </div>
      <div class="divider"></div>

      <h3 class="section-header">Property Input Specifications</h3>
      <div class="grid-3">
        <div>
          <div style="font-weight: 700; margin-bottom: 10px; color: #cbd5e1;">Structure &amp; Area</div>
          <div class="form-group">
            <label>Overall Quality (1: Poor, 10: Excellent)</label>
            <div class="slider-container">
              <input type="range" id="p1_qual" min="1" max="10" value="7" oninput="document.getElementById('p1_qual_val').innerText = this.value">
              <span id="p1_qual_val" class="slider-val">7</span>
            </div>
          </div>
          <div class="form-group">
            <label>Above Ground Living Area (sq ft)</label>
            <input type="number" id="p1_area" value="1750" step="25">
          </div>
          <div class="form-group">
            <label>Year Built</label>
            <input type="number" id="p1_year" value="2005" step="1">
          </div>
        </div>

        <div>
          <div style="font-weight: 700; margin-bottom: 10px; color: #cbd5e1;">Basement &amp; Amenities</div>
          <div class="form-group">
            <label>Basement Area (sq ft)</label>
            <input type="number" id="p1_bsmt" value="950" step="25">
          </div>
          <div class="form-group">
            <label>Garage Car Capacity</label>
            <select id="p1_garage">
              <option value="0">0</option>
              <option value="1">1</option>
              <option value="2" selected>2</option>
              <option value="3">3</option>
              <option value="4">4</option>
            </select>
          </div>
          <div class="form-group">
            <label>Full Bathrooms</label>
            <select id="p1_bath">
              <option value="1">1</option>
              <option value="2" selected>2</option>
              <option value="3">3</option>
              <option value="4">4</option>
            </select>
          </div>
        </div>

        <div>
          <div style="font-weight: 700; margin-bottom: 10px; color: #cbd5e1;">Location &amp; Finishes</div>
          <div class="form-group">
            <label>Neighborhood</label>
            <select id="p1_neigh">
              <option value="CollgCr">CollgCr</option>
              <option value="Veenker">Veenker</option>
              <option value="Crawfor">Crawfor</option>
              <option value="NridgHt">NridgHt</option>
              <option value="Mitchel">Mitchel</option>
              <option value="Somerst">Somerst</option>
              <option value="OldTown">OldTown</option>
            </select>
          </div>
          <div class="form-group">
            <label>Exterior Material Quality</label>
            <select id="p1_ext">
              <option value="Ex">Ex</option>
              <option value="Gd" selected>Gd</option>
              <option value="TA">TA</option>
              <option value="Fa">Fa</option>
            </select>
          </div>
          <div class="form-group">
            <label>Kitchen Quality</label>
            <select id="p1_kit">
              <option value="Ex">Ex</option>
              <option value="Gd" selected>Gd</option>
              <option value="TA">TA</option>
              <option value="Fa">Fa</option>
            </select>
          </div>
        </div>
      </div>

      <button class="st-btn" onclick="predictPrice()">Calculate Predicted Price (Linear Regression)</button>
      <div id="p1_alert" class="st-alert"></div>
    </div>

    <!-- PILLAR 2: CLASSIFICATION -->
    <div id="pillar2" class="pillar-pane">
      <h1 class="page-title">🎯 Classification Algorithms Suite</h1>
      <p class="page-desc">Predict customer churn probability across 5 benchmark classification models on the <strong>Telco Dataset</strong>.</p>
      <div class="divider"></div>

      <div class="form-group" style="max-width: 400px; margin-bottom: 20px;">
        <label>Choose Classification Algorithm:</label>
        <select id="p2_model">
          <option value="Logistic Regression">Logistic Regression</option>
          <option value="Decision Tree">Decision Tree</option>
          <option value="Random Forest" selected>Random Forest</option>
          <option value="Support Vector Machines">Support Vector Machines</option>
          <option value="Naive Bayes">Naive Bayes</option>
        </select>
      </div>

      <h3 class="section-header">Model Performance Comparison (Validation Set)</h3>
      <table>
        <thead>
          <tr>
            <th>Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th>
          </tr>
        </thead>
        <tbody id="p2_tbody">
          <tr><td colspan="6" style="color: var(--text-muted);">Loading benchmark metrics...</td></tr>
        </tbody>
      </table>
      <div class="divider"></div>

      <h3 class="section-header">Customer Features &amp; Inference</h3>
      <div class="grid-2">
        <div>
          <div class="form-group">
            <label>Tenure (Months with Provider)</label>
            <div class="slider-container">
              <input type="range" id="p2_tenure" min="1" max="72" value="12" oninput="document.getElementById('p2_tenure_val').innerText = this.value">
              <span id="p2_tenure_val" class="slider-val">12</span>
            </div>
          </div>
          <div class="form-group">
            <label>Monthly Charges ($)</label>
            <input type="number" id="p2_monthly" value="70.0" step="1.0">
          </div>
          <div class="form-group">
            <label>Total Charges ($)</label>
            <input type="number" id="p2_total" value="840.0" step="10.0">
          </div>
          <div class="form-group">
            <label>Contract Type</label>
            <select id="p2_contract">
              <option value="Month-to-month">Month-to-month</option>
              <option value="One year">One year</option>
              <option value="Two year">Two year</option>
            </select>
          </div>
        </div>

        <div>
          <div class="form-group">
            <label>Internet Service</label>
            <select id="p2_internet">
              <option value="DSL">DSL</option>
              <option value="Fiber optic" selected>Fiber optic</option>
              <option value="No">No</option>
            </select>
          </div>
          <div class="form-group">
            <label>Payment Method</label>
            <select id="p2_payment">
              <option value="Electronic check" selected>Electronic check</option>
              <option value="Mailed check">Mailed check</option>
              <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
              <option value="Credit card (automatic)">Credit card (automatic)</option>
            </select>
          </div>
          <div class="form-group">
            <label>Online Security</label>
            <select id="p2_security">
              <option value="No" selected>No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
          <div class="form-group">
            <label>Tech Support</label>
            <select id="p2_tech">
              <option value="No" selected>No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
        </div>
      </div>

      <button class="st-btn" onclick="predictChurn()">Run Prediction</button>
      <div id="p2_alert" class="st-alert"></div>
    </div>

    <!-- PILLAR 3: CLUSTERING -->
    <div id="pillar3" class="pillar-pane">
      <h1 class="page-title">🧩 Unsupervised Clustering Suite</h1>
      <p class="page-desc">Group customer segments on the <strong>Mall Customers Dataset</strong> without pre-existing labels.</p>
      <div class="divider"></div>

      <div class="form-group" style="max-width: 400px; margin-bottom: 20px;">
        <label>Choose Clustering Algorithm:</label>
        <select id="p3_algo" onchange="renderClusteringChart()">
          <option value="K-Means">K-Means</option>
          <option value="DBSCAN">DBSCAN</option>
          <option value="Hierarchical Clustering">Hierarchical Clustering</option>
        </select>
      </div>

      <h3 class="section-header">Cluster Separation Metrics (Silhouette Scores)</h3>
      <table>
        <thead>
          <tr><th>Algorithm</th><th>Silhouette Score</th><th>Parameters</th></tr>
        </thead>
        <tbody id="p3_tbody">
          <tr><td colspan="3" style="color: var(--text-muted);">Loading clustering metrics...</td></tr>
        </tbody>
      </table>

      <h3 class="section-header" id="p3_chart_title">Cluster Visualization: K-Means</h3>
      <div class="chart-box">
        <canvas id="clusterChart"></canvas>
      </div>
    </div>

    <!-- PILLAR 4: DIM REDUCTION -->
    <div id="pillar4" class="pillar-pane">
      <h1 class="page-title">📉 Dimensionality Reduction Suite</h1>
      <p class="page-desc">Compress multi-dimensional customer feature space into 2D projections using <strong>PCA</strong> and <strong>t-SNE</strong>.</p>
      <div class="divider"></div>

      <div class="form-group" style="max-width: 400px; margin-bottom: 20px;">
        <label>Choose Dimensionality Reduction Technique:</label>
        <select id="p4_algo" onchange="renderDimChart()">
          <option value="PCA">PCA (Principal Component Analysis)</option>
          <option value="tSNE">t-SNE (t-Distributed Stochastic Neighbor Embedding)</option>
        </select>
      </div>

      <div id="p4_alert" class="st-alert info">Loading projection details...</div>

      <div class="chart-box">
        <canvas id="dimChart"></canvas>
      </div>
    </div>

  </div>

  <script>
    let clusteringData = null;
    let dimData = null;
    let clusterChart = null;
    let dimChart = null;

    function switchPillar(pillarId, element) {
      document.querySelectorAll('.radio-option').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.pillar-pane').forEach(el => el.classList.remove('active'));
      element.classList.add('active');
      document.getElementById(pillarId).classList.add('active');
      
      if (pillarId === 'pillar3' && !clusteringData) loadClusteringData();
      if (pillarId === 'pillar4' && !dimData) loadDimData();
    }

    // PILLAR 1: PREDICT PRICE
    async function predictPrice() {
      const payload = {
        OverallQual: parseInt(document.getElementById("p1_qual").value),
        GrLivArea: parseFloat(document.getElementById("p1_area").value),
        TotalBsmtSF: parseFloat(document.getElementById("p1_bsmt").value),
        GarageCars: parseInt(document.getElementById("p1_garage").value),
        FullBath: parseInt(document.getElementById("p1_bath").value),
        YearBuilt: parseInt(document.getElementById("p1_year").value),
        Neighborhood: document.getElementById("p1_neigh").value,
        ExterQual: document.getElementById("p1_ext").value,
        KitchenQual: document.getElementById("p1_kit").value,
      };

      const box = document.getElementById("p1_alert");
      box.className = "st-alert info";
      box.innerText = "⏳ Calculating valuation...";
      box.style.display = "block";

      try {
        const res = await fetch("/api/predict-price", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: "Server error" }));
          throw new Error(err.detail || "HTTP " + res.status);
        }
        const data = await res.json();
        const price = parseFloat(data.predicted_price);
        box.className = "st-alert success";
        box.innerText = "Estimated Property Valuation: $" + price.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2});
      } catch (err) {
        box.className = "st-alert error";
        box.innerText = "Error: " + err.message;
      }
    }

    // PILLAR 2: CLASSIFICATION
    async function loadClassificationMetrics() {
      const tbody = document.getElementById("p2_tbody");
      try {
        const res = await fetch("/api/classification-metrics");
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();
        tbody.innerHTML = "";
        data.metrics.forEach(row => {
          tbody.innerHTML += `<tr>
            <td style="font-weight:600; color:var(--text-main);">${row["Algorithm"]}</td>
            <td>${(row["Accuracy"]*100).toFixed(1)}%</td>
            <td>${(row["Precision"]*100).toFixed(1)}%</td>
            <td>${(row["Recall"]*100).toFixed(1)}%</td>
            <td>${parseFloat(row["F1-Score"]).toFixed(3)}</td>
            <td><strong style="color:var(--success-green);">${parseFloat(row["ROC-AUC"]).toFixed(3)}</strong></td>
          </tr>`;
        });
      } catch (e) {
        tbody.innerHTML = `<tr><td colspan="6" style="color:var(--error-red);">Could not load metrics: ${e.message}</td></tr>`;
      }
    }

    async function predictChurn() {
      const selectedModel = document.getElementById("p2_model").value;
      const payload = {
        tenure: parseInt(document.getElementById("p2_tenure").value),
        MonthlyCharges: parseFloat(document.getElementById("p2_monthly").value),
        TotalCharges: parseFloat(document.getElementById("p2_total").value),
        Contract: document.getElementById("p2_contract").value,
        InternetService: document.getElementById("p2_internet").value,
        PaymentMethod: document.getElementById("p2_payment").value,
        OnlineSecurity: document.getElementById("p2_security").value,
        TechSupport: document.getElementById("p2_tech").value,
        Algorithm: selectedModel
      };

      const box = document.getElementById("p2_alert");
      box.className = "st-alert info";
      box.innerText = "⏳ Running inference...";
      box.style.display = "block";

      try {
        const res = await fetch("/api/predict-churn", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: "Server error" }));
          throw new Error(err.detail || "HTTP " + res.status);
        }
        const data = await res.json();
        const prob = parseFloat(data.churn_probability);
        const pred = parseInt(data.churn_prediction);
        if (pred === 1) {
          box.className = "st-alert error";
          box.innerText = `Prediction: Churn Risk Detected (Probability: ${(prob * 100).toFixed(1)}%)`;
        } else {
          box.className = "st-alert success";
          box.innerText = `Prediction: Loyal Customer (Probability of Staying: ${((1 - prob) * 100).toFixed(1)}%)`;
        }
      } catch (err) {
        box.className = "st-alert error";
        box.innerText = "Error: " + err.message;
      }
    }

    // PILLAR 3: CLUSTERING
    async function loadClusteringData() {
      const tbody = document.getElementById("p3_tbody");
      try {
        const res = await fetch("/api/clustering-data");
        if (!res.ok) throw new Error("HTTP " + res.status);
        clusteringData = await res.json();
        tbody.innerHTML = "";
        clusteringData.metrics.forEach(row => {
          tbody.innerHTML += `<tr>
            <td style="font-weight:600;">${row["Algorithm"]}</td>
            <td><strong style="color:var(--success-green);">${parseFloat(row["Silhouette Score"]).toFixed(4)}</strong></td>
            <td style="color:var(--text-muted);">${row["Parameters"]}</td>
          </tr>`;
        });
        renderClusteringChart();
      } catch (e) {
        tbody.innerHTML = `<tr><td colspan="3" style="color:var(--error-red);">Could not load metrics: ${e.message}</td></tr>`;
      }
    }

    function renderClusteringChart() {
      if (!clusteringData) return;
      const algo = document.getElementById("p3_algo").value;
      document.getElementById("p3_chart_title").innerText = `Cluster Visualization: ${algo}`;
      const colMap = {
        "K-Means": "KMeans_Cluster",
        "DBSCAN": "DBSCAN_Cluster",
        "Hierarchical Clustering": "Hierarchical_Cluster"
      };
      const key = colMap[algo];
      const colors = ['#ff4b4b', '#58a6ff', '#21c354', '#f59e0b', '#a855f7', '#ec4899'];
      const datasets = {};

      clusteringData.plot.forEach(pt => {
        const cId = String(pt[key]);
        if (!datasets[cId]) {
          datasets[cId] = {
            label: `Cluster ${cId}`,
            data: [],
            backgroundColor: colors[Math.abs(parseInt(cId) || 0) % colors.length],
            pointRadius: 6, pointHoverRadius: 9
          };
        }
        datasets[cId].data.push({ x: pt["Annual Income (k$)"], y: pt["Spending Score (1-100)"] });
      });

      const ctx = document.getElementById("clusterChart").getContext("2d");
      if (clusterChart) clusterChart.destroy();
      clusterChart = new Chart(ctx, {
        type: "scatter",
        data: { datasets: Object.values(datasets) },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: "Annual Income (k$)", color: "#a3a8b8" }, grid: { color: "#31333f" }, ticks: { color: "#fafafa" } },
            y: { title: { display: true, text: "Spending Score (1-100)", color: "#a3a8b8" }, grid: { color: "#31333f" }, ticks: { color: "#fafafa" } }
          },
          plugins: { legend: { labels: { color: "#fafafa" } } }
        }
      });
    }

    // PILLAR 4: DIM REDUCTION
    async function loadDimData() {
      const box = document.getElementById("p4_alert");
      try {
        const res = await fetch("/api/dim-reduction-data");
        if (!res.ok) throw new Error("HTTP " + res.status);
        dimData = await res.json();
        renderDimChart();
      } catch (e) {
        box.className = "st-alert error";
        box.innerText = "Could not load data: " + e.message;
      }
    }

    function renderDimChart() {
      if (!dimData) return;
      const algo = document.getElementById("p4_algo").value;
      const box = document.getElementById("p4_alert");
      let xKey, yKey;
      if (algo === "PCA") {
        xKey = "PCA_1"; yKey = "PCA_2";
        const tv = (dimData.metrics.PCA_Total_Var * 100).toFixed(2);
        const pc1 = (dimData.metrics.PCA_PC1_Var * 100).toFixed(1);
        const pc2 = (dimData.metrics.PCA_PC2_Var * 100).toFixed(1);
        box.className = "st-alert info";
        box.innerText = `Total Information Retained in 2D: ${tv}% (PC1: ${pc1}%, PC2: ${pc2}%)`;
      } else {
        xKey = "tSNE_1"; yKey = "tSNE_2";
        const kl = parseFloat(dimData.metrics.tSNE_KL_Divergence).toFixed(4);
        box.className = "st-alert info";
        box.innerText = `Non-linear Neighborhood Embedding — Final KL Divergence: ${kl}`;
      }

      const points = dimData.dim_df.map(pt => ({ x: parseFloat(pt[xKey]), y: parseFloat(pt[yKey]) }));
      const ctx = document.getElementById("dimChart").getContext("2d");
      if (dimChart) dimChart.destroy();
      dimChart = new Chart(ctx, {
        type: "scatter",
        data: {
          datasets: [{
            label: `${algo} 2D Projection`,
            data: points,
            backgroundColor: "#ff4b4b",
            pointRadius: 5, pointHoverRadius: 8
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: `${algo} — Dimension 1`, color: "#a3a8b8" }, grid: { color: "#31333f" }, ticks: { color: "#fafafa" } },
            y: { title: { display: true, text: `${algo} — Dimension 2`, color: "#a3a8b8" }, grid: { color: "#31333f" }, ticks: { color: "#fafafa" } }
          },
          plugins: { legend: { labels: { color: "#fafafa" } } }
        }
      });
    }

    // ON PAGE LOAD
    document.addEventListener("DOMContentLoaded", () => {
      loadClassificationMetrics();
      loadClusteringData();
      loadDimData();
    });
  </script>
</body>
</html>"""


# ==============================================================================
# HEALTH & UPTIME ENDPOINTS
# ==============================================================================
@app.get("/health")
def health_check():
    uptime_seconds = int(time.time() - APP_START_TIME)
    return {
        "status": "healthy",
        "uptime_seconds": uptime_seconds,
        "uptime_hours": round(uptime_seconds / 3600, 2),
        "app": "ML Master Suite",
        "version": "2.0.0"
    }

@app.get("/ping")
def ping():
    return {"pong": True, "timestamp": int(time.time())}


# ==============================================================================
# FASTAPI PREDICTION ENDPOINTS
# ==============================================================================
@app.post("/api/predict-price")
def predict_price(features: HouseFeatures):
    try:
        artifacts = get_artifact("linear_house_price_pipeline.joblib")
        pipeline = artifacts["pipeline"]
        data_dict = features.model_dump() if hasattr(features, "model_dump") else features.dict()
        input_df = pd.DataFrame([data_dict])
        log_pred = pipeline.predict(input_df)[0]
        final_price = float(np.expm1(log_pred))
        return {
            "predicted_price": round(final_price, 2),
            "metrics": artifacts.get("metrics", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/classification-metrics")
def get_classification_metrics():
    try:
        artifacts = get_artifact("classification_suite.joblib")
        metrics_df = artifacts["metrics_df"]
        return {"metrics": metrics_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict-churn")
def predict_churn(features: ChurnFeatures):
    try:
        artifacts = get_artifact("classification_suite.joblib")
        pipelines = artifacts["pipelines"]
        
        algo_name = features.Algorithm
        if algo_name not in pipelines:
            algo_name = "Random Forest"
            
        pipeline = pipelines[algo_name]
        data_dict = features.model_dump() if hasattr(features, "model_dump") else features.dict()
        if "Algorithm" in data_dict:
            del data_dict["Algorithm"]
        
        input_df = pd.DataFrame([data_dict])
        pred = int(pipeline.predict(input_df)[0])
        probs = pipeline.predict_proba(input_df)[0]
        churn_prob = float(probs[1]) if len(probs) > 1 else float(pred)
        
        return {
            "model": algo_name,
            "churn_prediction": pred,
            "churn_probability": round(churn_prob, 4),
            "message": "Churn risk detected" if pred == 1 else "Loyal customer"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clustering-data")
def get_clustering_data():
    try:
        artifacts = get_artifact("clustering_suite.joblib")
        metrics_df = artifacts["metrics_df"]
        plot_df = artifacts["plot_df"]
        return {
            "metrics": metrics_df.to_dict(orient="records"),
            "plot": plot_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dim-reduction-data")
def get_dim_reduction_data():
    try:
        artifacts = get_artifact("dim_reduction_suite.joblib")
        metrics_info = artifacts["metrics_info"]
        dim_df = artifacts["dim_df"]
        # Convert float64 numpy to standard python float for clean json serialization
        clean_metrics = {k: float(v) for k, v in metrics_info.items()}
        return {
            "metrics": clean_metrics,
            "dim_df": dim_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))