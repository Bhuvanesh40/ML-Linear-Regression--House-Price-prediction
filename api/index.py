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
# HTML & JS FRONTEND DASHBOARD
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Machine Learning Master Suite (11 Algorithms)</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg-color: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-indigo: #818cf8;
      --accent-green: #34d399;
      --accent-rose: #fb7185;
      --btn-bg: #2563eb;
      --btn-hover: #1d4ed8;
    }
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 24px;
      background: var(--bg-color);
      color: var(--text-main);
      min-height: 100vh;
    }
    .header {
      text-align: center;
      margin-bottom: 28px;
    }
    .header h1 {
      margin: 0;
      font-size: 2rem;
      color: var(--accent-blue);
      letter-spacing: -0.02em;
    }
    .header p {
      margin-top: 8px;
      color: var(--text-muted);
      font-size: 0.95rem;
    }
    .nav-tabs {
      display: flex;
      justify-content: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 24px;
    }
    .tab-btn {
      background: var(--card-bg);
      color: var(--text-muted);
      border: 1px solid var(--card-border);
      padding: 10px 18px;
      border-radius: 9999px;
      font-weight: 600;
      font-size: 0.88rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .tab-btn:hover { background: #334155; color: var(--text-main); }
    .tab-btn.active {
      background: var(--btn-bg);
      color: white;
      border-color: var(--btn-bg);
      box-shadow: 0 0 12px rgba(37, 99, 235, 0.4);
    }
    .container {
      max-width: 960px;
      margin: 0 auto;
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .card {
      background: var(--card-bg);
      padding: 24px;
      border-radius: 14px;
      border: 1px solid var(--card-border);
      margin-bottom: 20px;
    }
    h2 { margin-top: 0; color: var(--accent-blue); font-size: 1.35rem; }
    h3 { margin-top: 16px; margin-bottom: 8px; font-size: 1.1rem; color: var(--accent-indigo); }
    .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
    .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    label { display: block; margin-top: 10px; font-weight: 600; font-size: 0.82rem; color: #cbd5e1; }
    input, select {
      width: 100%;
      padding: 10px 12px;
      margin-top: 5px;
      border-radius: 8px;
      border: 1px solid #475569;
      background: var(--bg-color);
      color: var(--text-main);
      font-size: 0.9rem;
    }
    input:focus, select:focus { outline: none; border-color: var(--accent-blue); }
    button.submit-btn {
      width: 100%;
      margin-top: 20px;
      padding: 12px;
      border: none;
      border-radius: 8px;
      background: var(--btn-bg);
      color: white;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
      transition: background 0.2s ease;
    }
    button.submit-btn:hover { background: var(--btn-hover); }
    .result-box {
      margin-top: 18px;
      padding: 14px 18px;
      border-radius: 10px;
      display: none;
      text-align: center;
      font-weight: bold;
      font-size: 1.05rem;
    }
    .metrics-row {
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }
    .metric-badge {
      background: #0f172a;
      border: 1px solid var(--card-border);
      padding: 10px 16px;
      border-radius: 8px;
      flex: 1;
      min-width: 120px;
      text-align: center;
    }
    .metric-badge span { display: block; font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; }
    .metric-badge strong { font-size: 1.1rem; color: var(--accent-green); }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.88rem; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid var(--card-border); }
    th { color: var(--accent-blue); background: #0f172a; }
    .chart-container { position: relative; height: 380px; width: 100%; margin-top: 16px; }
  </style>
</head>
<body>

  <div class="header">
    <h1>🤖 Machine Learning Master Suite</h1>
    <p>11 Core Machine Learning Algorithms Across 4 Interactive Pillars</p>
  </div>

  <div class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('tab1', event)">🏡 1. House Price Regression</button>
    <button class="tab-btn" onclick="switchTab('tab2', event)">🎯 2. Churn Classification (5 Models)</button>
    <button class="tab-btn" onclick="switchTab('tab3', event)">🧩 3. Unsupervised Clustering (3 Models)</button>
    <button class="tab-btn" onclick="switchTab('tab4', event)">📉 4. Dim Reduction (PCA &amp; t-SNE)</button>
  </div>

  <div class="container">

    <!-- PILLAR 1: REGRESSION -->
    <div id="tab1" class="tab-content active">
      <div class="card">
        <h2>🏡 House Price Estimator (Ridge Regularized Linear Regression)</h2>
        <p style="color: var(--text-muted); font-size: 0.88rem;">Predict property values on the Ames Housing Dataset using regularized linear regression.</p>
        
        <div class="metrics-row">
          <div class="metric-badge"><span>Validation R²</span><strong>0.8720</strong></div>
          <div class="metric-badge"><span>Validation RMSLE</span><strong>0.1348</strong></div>
          <div class="metric-badge"><span>Validation MAE</span><strong>$18,635.80</strong></div>
        </div>

        <div class="grid-3">
          <div>
            <label>Overall Quality (1-10)</label>
            <input type="number" id="p1_qual" value="7" min="1" max="10" />
            <label>Above Ground Living Area (sq ft)</label>
            <input type="number" id="p1_area" value="1750" />
            <label>Year Built</label>
            <input type="number" id="p1_year" value="2005" />
          </div>
          <div>
            <label>Basement Area (sq ft)</label>
            <input type="number" id="p1_bsmt" value="950" />
            <label>Garage Car Capacity</label>
            <select id="p1_garage">
              <option value="0">0 Cars</option>
              <option value="1">1 Car</option>
              <option value="2" selected>2 Cars</option>
              <option value="3">3 Cars</option>
            </select>
            <label>Full Bathrooms</label>
            <select id="p1_bath">
              <option value="1">1</option>
              <option value="2" selected>2</option>
              <option value="3">3</option>
            </select>
          </div>
          <div>
            <label>Neighborhood</label>
            <select id="p1_neigh">
              <option value="CollgCr">College Creek (CollgCr)</option>
              <option value="Veenker">Veenker</option>
              <option value="Crawfor">Crawford</option>
              <option value="NridgHt">Northridge Heights</option>
              <option value="Somerst">Somerset</option>
              <option value="OldTown">Old Town</option>
            </select>
            <label>Exterior Quality</label>
            <select id="p1_ext">
              <option value="Ex">Excellent</option>
              <option value="Gd" selected>Good</option>
              <option value="TA">Average</option>
              <option value="Fa">Fair</option>
            </select>
            <label>Kitchen Quality</label>
            <select id="p1_kit">
              <option value="Ex">Excellent</option>
              <option value="Gd" selected>Good</option>
              <option value="TA">Average</option>
              <option value="Fa">Fair</option>
            </select>
          </div>
        </div>

        <button class="submit-btn" onclick="predictPrice()">Calculate Property Valuation</button>
        <div id="p1_result" class="result-box"></div>
      </div>
    </div>


    <!-- PILLAR 2: CLASSIFICATION -->
    <div id="tab2" class="tab-content">
      <div class="card">
        <h2>🎯 Telecom Customer Churn Classifier</h2>
        <p style="color: var(--text-muted); font-size: 0.88rem;">Compare and run predictions across 5 benchmark classification algorithms on the Telco Churn dataset.</p>
        
        <label style="font-size: 0.9rem; color: var(--accent-blue);">Choose Classification Model:</label>
        <select id="p2_model" style="margin-bottom: 16px;">
          <option value="Logistic Regression">Logistic Regression (Probabilistic Sigmoid)</option>
          <option value="Decision Tree">Decision Tree (Information Gain Splits)</option>
          <option value="Random Forest" selected>Random Forest (150-Tree Ensemble)</option>
          <option value="Support Vector Machines">Support Vector Machines (RBF Kernel Margin)</option>
          <option value="Naive Bayes">Gaussian Naive Bayes (Bayesian Probability)</option>
        </select>

        <h3>Benchmark Performance Comparison</h3>
        <div style="overflow-x: auto;">
          <table>
            <thead>
              <tr>
                <th>Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th>
              </tr>
            </thead>
            <tbody id="p2_metrics_body">
              <tr><td colspan="6">Loading benchmark metrics...</td></tr>
            </tbody>
          </table>
        </div>

        <h3>Customer Features & Inference</h3>
        <div class="grid-2">
          <div>
            <label>Tenure (Months with Provider)</label>
            <input type="number" id="p2_tenure" value="12" min="1" max="72" />
            <label>Monthly Charges ($)</label>
            <input type="number" id="p2_monthly" value="70.0" step="1.0" />
            <label>Total Charges ($)</label>
            <input type="number" id="p2_total" value="840.0" step="10.0" />
            <label>Contract Type</label>
            <select id="p2_contract">
              <option value="Month-to-month">Month-to-month</option>
              <option value="One year">One year</option>
              <option value="Two year">Two year</option>
            </select>
          </div>
          <div>
            <label>Internet Service</label>
            <select id="p2_internet">
              <option value="DSL">DSL</option>
              <option value="Fiber optic" selected>Fiber optic</option>
              <option value="No">No</option>
            </select>
            <label>Payment Method</label>
            <select id="p2_payment">
              <option value="Electronic check" selected>Electronic check</option>
              <option value="Mailed check">Mailed check</option>
              <option value="Bank transfer (automatic)">Bank transfer</option>
              <option value="Credit card (automatic)">Credit card</option>
            </select>
            <label>Online Security</label>
            <select id="p2_security">
              <option value="No" selected>No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
            <label>Tech Support</label>
            <select id="p2_tech">
              <option value="No" selected>No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
        </div>

        <button class="submit-btn" onclick="predictChurn()">Run Churn Prediction</button>
        <div id="p2_result" class="result-box"></div>
      </div>
    </div>


    <!-- PILLAR 3: CLUSTERING -->
    <div id="tab3" class="tab-content">
      <div class="card">
        <h2>🧩 Unsupervised Customer Clustering</h2>
        <p style="color: var(--text-muted); font-size: 0.88rem;">Segment mall shoppers into behavioral cohorts without pre-existing labels.</p>
        
        <label style="font-size: 0.9rem; color: var(--accent-blue);">Select Clustering Algorithm:</label>
        <select id="p3_algo" onchange="renderClusteringChart()" style="margin-bottom: 16px;">
          <option value="K-Means">K-Means (k=5 Centroids)</option>
          <option value="DBSCAN">DBSCAN (Density-Based Spatial Clustering)</option>
          <option value="Hierarchical Clustering">Hierarchical (Agglomerative Linkage)</option>
        </select>

        <h3>Silhouette Scores & Parameters</h3>
        <div style="overflow-x: auto;">
          <table>
            <thead>
              <tr><th>Algorithm</th><th>Silhouette Score</th><th>Parameters</th></tr>
            </thead>
            <tbody id="p3_metrics_body">
              <tr><td colspan="3">Loading clustering evaluation...</td></tr>
            </tbody>
          </table>
        </div>

        <h3>Interactive Customer Cluster Scatter Plot</h3>
        <div class="chart-container">
          <canvas id="clusteringChart"></canvas>
        </div>
      </div>
    </div>


    <!-- PILLAR 4: DIMENSIONALITY REDUCTION -->
    <div id="tab4" class="tab-content">
      <div class="card">
        <h2>📉 Dimensionality Reduction (PCA & t-SNE)</h2>
        <p style="color: var(--text-muted); font-size: 0.88rem;">Compress high-dimensional customer space into 2D projections for visual analysis.</p>

        <label style="font-size: 0.9rem; color: var(--accent-blue);">Select Technique:</label>
        <select id="p4_algo" onchange="renderDimChart()" style="margin-bottom: 16px;">
          <option value="PCA">PCA (Principal Component Analysis - Linear Variance)</option>
          <option value="tSNE">t-SNE (t-Distributed Stochastic Neighbor Embedding - Non-linear)</option>
        </select>

        <div id="p4_info" class="metric-badge" style="text-align: left; margin-bottom: 16px; padding: 14px;">
          <span>Technique Information</span>
          <strong id="p4_info_text" style="font-size: 0.95rem; color: var(--accent-blue);">Loading info...</strong>
        </div>

        <h3>2D Manifold Projection Chart</h3>
        <div class="chart-container">
          <canvas id="dimChart"></canvas>
        </div>
      </div>
    </div>

  </div>

  <script>
    let clusteringData = null;
    let dimData = null;
    let clusterChartInstance = null;
    let dimChartInstance = null;

    function switchTab(tabId, evt) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

      if (evt && evt.currentTarget) evt.currentTarget.classList.add('active');
      document.getElementById(tabId).classList.add('active');

      if (tabId === 'tab3' && !clusteringData) loadClusteringData();
      if (tabId === 'tab4' && !dimData) loadDimData();
    }

    // ─────────────────────────────────────────────────────────────────
    // PILLAR 1: REGRESSION — House Price Prediction
    // ─────────────────────────────────────────────────────────────────
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

      const box = document.getElementById("p1_result");
      box.style.display = "block";
      box.style.background = "#1e293b";
      box.style.color = "#94a3b8";
      box.innerText = "⏳ Calculating valuation...";

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
        if (isNaN(price)) throw new Error("Invalid price response from server.");
        box.style.background = "#14532d";
        box.style.color = "#86efac";
        box.innerText = "🏡 Estimated Valuation: $" + price.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2});
      } catch (err) {
        box.style.background = "#7f1d1d";
        box.style.color = "#fca5a5";
        box.innerText = "❌ Error: " + err.message;
      }
    }

    // ─────────────────────────────────────────────────────────────────
    // PILLAR 2: CLASSIFICATION — Metrics + Churn Prediction
    // ─────────────────────────────────────────────────────────────────
    async function loadClassificationMetrics() {
      const tbody = document.getElementById("p2_metrics_body");
      tbody.innerHTML = "<tr><td colspan='6' style='color:#94a3b8;'>Loading benchmark metrics...</td></tr>";
      try {
        const res = await fetch("/api/classification-metrics");
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();
        tbody.innerHTML = "";
        data.metrics.forEach(row => {
          const acc  = (row["Accuracy"]  * 100).toFixed(1);
          const prec = (row["Precision"] * 100).toFixed(1);
          const rec  = (row["Recall"]    * 100).toFixed(1);
          const f1   = parseFloat(row["F1-Score"]).toFixed(3);
          const auc  = parseFloat(row["ROC-AUC"]).toFixed(3);
          tbody.innerHTML += `<tr>
            <td style="font-weight:600; color:var(--text-main);">${row["Algorithm"]}</td>
            <td>${acc}%</td><td>${prec}%</td><td>${rec}%</td>
            <td>${f1}</td>
            <td><strong style="color:var(--accent-green);">${auc}</strong></td>
          </tr>`;
        });
      } catch(e) {
        tbody.innerHTML = `<tr><td colspan='6' style='color:#fca5a5;'>⚠ Could not load metrics: ${e.message}</td></tr>`;
      }
    }

    async function predictChurn() {
      const payload = {
        tenure: parseInt(document.getElementById("p2_tenure").value),
        MonthlyCharges: parseFloat(document.getElementById("p2_monthly").value),
        TotalCharges: parseFloat(document.getElementById("p2_total").value),
        Contract: document.getElementById("p2_contract").value,
        InternetService: document.getElementById("p2_internet").value,
        PaymentMethod: document.getElementById("p2_payment").value,
        OnlineSecurity: document.getElementById("p2_security").value,
        TechSupport: document.getElementById("p2_tech").value,
        Algorithm: document.getElementById("p2_model").value
      };

      const box = document.getElementById("p2_result");
      box.style.display = "block";
      box.style.background = "#1e293b";
      box.style.color = "#94a3b8";
      box.innerText = "⏳ Running classifier inference...";

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
        const prob      = parseFloat(data.churn_probability);
        const pred      = parseInt(data.churn_prediction);
        const modelName = data.model || payload.Algorithm;
        if (isNaN(prob)) throw new Error("Invalid probability in server response.");
        if (pred === 1) {
          box.style.background = "#7f1d1d";
          box.style.color = "#fca5a5";
          box.innerText = "🚨 Churn Risk Detected — Probability: " + (prob * 100).toFixed(1) + "%   |   Model: " + modelName;
        } else {
          box.style.background = "#14532d";
          box.style.color = "#86efac";
          box.innerText = "✅ Loyal Customer — Probability of Staying: " + ((1 - prob) * 100).toFixed(1) + "%   |   Model: " + modelName;
        }
      } catch (err) {
        box.style.background = "#7f1d1d";
        box.style.color = "#fca5a5";
        box.innerText = "❌ Error: " + err.message;
      }
    }

    // ─────────────────────────────────────────────────────────────────
    // PILLAR 3: CLUSTERING
    // ─────────────────────────────────────────────────────────────────
    async function loadClusteringData() {
      const tbody = document.getElementById("p3_metrics_body");
      tbody.innerHTML = "<tr><td colspan='3' style='color:#94a3b8;'>Loading clustering evaluation...</td></tr>";
      try {
        const res = await fetch("/api/clustering-data");
        if (!res.ok) throw new Error("HTTP " + res.status);
        clusteringData = await res.json();
        tbody.innerHTML = "";
        clusteringData.metrics.forEach(row => {
          tbody.innerHTML += `<tr>
            <td style="font-weight:600;">${row["Algorithm"]}</td>
            <td><strong style="color:var(--accent-green);">${parseFloat(row["Silhouette Score"]).toFixed(4)}</strong></td>
            <td style="color:var(--text-muted);">${row["Parameters"]}</td>
          </tr>`;
        });
        renderClusteringChart();
      } catch(e) {
        tbody.innerHTML = `<tr><td colspan='3' style='color:#fca5a5;'>⚠ Could not load data: ${e.message}</td></tr>`;
      }
    }

    function renderClusteringChart() {
      if (!clusteringData) return;
      const selectedAlgo = document.getElementById("p3_algo").value;
      const colMap = {
        "K-Means": "KMeans_Cluster",
        "DBSCAN": "DBSCAN_Cluster",
        "Hierarchical Clustering": "Hierarchical_Cluster"
      };
      const key    = colMap[selectedAlgo];
      const colors = ['#38bdf8','#818cf8','#34d399','#fb7185','#facc15','#a78bfa'];
      const datasets = {};
      clusteringData.plot.forEach(pt => {
        const cId = String(pt[key]);
        if (!datasets[cId]) {
          datasets[cId] = {
            label: "Cluster " + cId,
            data: [],
            backgroundColor: colors[Math.abs(parseInt(cId) || 0) % colors.length],
            pointRadius: 6, pointHoverRadius: 9
          };
        }
        datasets[cId].data.push({ x: pt["Annual Income (k$)"], y: pt["Spending Score (1-100)"] });
      });

      const ctx = document.getElementById('clusteringChart').getContext('2d');
      if (clusterChartInstance) clusterChartInstance.destroy();
      clusterChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: { datasets: Object.values(datasets) },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: 'Annual Income (k$)', color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#cbd5e1' } },
            y: { title: { display: true, text: 'Spending Score (1-100)', color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#cbd5e1' } }
          },
          plugins: { legend: { labels: { color: '#f8fafc', padding: 16 } } }
        }
      });
    }

    // ─────────────────────────────────────────────────────────────────
    // PILLAR 4: DIMENSIONALITY REDUCTION
    // ─────────────────────────────────────────────────────────────────
    async function loadDimData() {
      document.getElementById("p4_info_text").innerText = "Loading projection data...";
      try {
        const res = await fetch("/api/dim-reduction-data");
        if (!res.ok) throw new Error("HTTP " + res.status);
        dimData = await res.json();
        renderDimChart();
      } catch(e) {
        document.getElementById("p4_info_text").innerText = "⚠ Could not load data: " + e.message;
      }
    }

    function renderDimChart() {
      if (!dimData) return;
      const algo     = document.getElementById("p4_algo").value;
      const infoText = document.getElementById("p4_info_text");
      let xKey, yKey;
      if (algo === 'PCA') {
        xKey = 'PCA_1'; yKey = 'PCA_2';
        const tv  = (dimData.metrics.PCA_Total_Var  * 100).toFixed(2);
        const pc1 = (dimData.metrics.PCA_PC1_Var    * 100).toFixed(1);
        const pc2 = (dimData.metrics.PCA_PC2_Var    * 100).toFixed(1);
        infoText.innerText = "Total Information Retained in 2D: " + tv + "%   (PC1: " + pc1 + "%,  PC2: " + pc2 + "%)";
      } else {
        xKey = 'tSNE_1'; yKey = 'tSNE_2';
        const kl = parseFloat(dimData.metrics.tSNE_KL_Divergence).toFixed(4);
        infoText.innerText = "Non-linear Neighborhood Embedding — Final KL Divergence: " + kl;
      }

      const points = dimData.dim_df
        .map(pt => ({ x: parseFloat(pt[xKey]), y: parseFloat(pt[yKey]) }))
        .filter(pt => !isNaN(pt.x) && !isNaN(pt.y));

      const ctx = document.getElementById('dimChart').getContext('2d');
      if (dimChartInstance) dimChartInstance.destroy();
      dimChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
          datasets: [{
            label: algo + " 2D Projection",
            data: points,
            backgroundColor: '#818cf8',
            pointRadius: 5, pointHoverRadius: 7
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: algo + " — Dimension 1", color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#cbd5e1' } },
            y: { title: { display: true, text: algo + " — Dimension 2", color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#cbd5e1' } }
          },
          plugins: { legend: { labels: { color: '#f8fafc' } } }
        }
      });
    }

    // ── INIT: pre-load ALL data immediately on page ready ──────────────
    document.addEventListener('DOMContentLoaded', function() {
      loadClassificationMetrics();
      loadClusteringData();
      loadDimData();
    });
  </script>
</body>
</html>"""


# ==============================================================================
# HEALTH & UPTIME ENDPOINTS (for 24/7 keep-alive monitoring)
# ==============================================================================
@app.get("/health")
def health_check():
    """Health check endpoint for Render.com and uptime monitors."""
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
    """Lightweight ping endpoint to prevent Render free-tier spin-down."""
    return {"pong": True, "timestamp": int(time.time())}


# ==============================================================================
# FASTAPI ENDPOINTS
# ==============================================================================

# Pillar 1 Endpoint
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


# Pillar 2 Endpoints
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


# Pillar 3 Endpoint
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


# Pillar 4 Endpoint
@app.get("/api/dim-reduction-data")
def get_dim_reduction_data():
    try:
        artifacts = get_artifact("dim_reduction_suite.joblib")
        metrics_info = artifacts["metrics_info"]
        dim_df = artifacts["dim_df"]
        return {
            "metrics": metrics_info,
            "dim_df": dim_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))