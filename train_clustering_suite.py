"""
# Clustering Algorithms Suite:
# 1. K-Means (Centroid-based partition minimizing within-cluster variance)
# 2. DBSCAN (Density-based spatial clustering identifying core samples and noise)
# 3. Hierarchical Clustering (Agglomerative bottom-up tree linkage clustering)
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Load Data
df = pd.read_csv("Mall_Customers.csv")
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# 2. Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Algorithm 1: K-Means (k=5)
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_sil = silhouette_score(X_scaled, kmeans_labels)

# 4. Algorithm 2: DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
# Filter noise (-1) for silhouette calculation if valid clusters exist
valid_mask = dbscan_labels != -1
dbscan_sil = silhouette_score(X_scaled[valid_mask], dbscan_labels[valid_mask]) if len(set(dbscan_labels[valid_mask])) > 1 else 0.0

# 5. Algorithm 3: Hierarchical (Agglomerative)
hierarchical = AgglomerativeClustering(n_clusters=5, linkage="ward")
hierarchical_labels = hierarchical.fit_predict(X_scaled)
hierarchical_sil = silhouette_score(X_scaled, hierarchical_labels)

metrics_df = pd.DataFrame([
    {"Algorithm": "K-Means", "Silhouette Score": kmeans_sil, "Parameters": "k=5, init='k-means++'"},
    {"Algorithm": "DBSCAN", "Silhouette Score": dbscan_sil, "Parameters": "eps=0.5, min_samples=5"},
    {"Algorithm": "Hierarchical Clustering", "Silhouette Score": hierarchical_sil, "Parameters": "k=5, linkage='ward'"}
])

print("=== Clustering Evaluation ===")
print(metrics_df.to_string(index=False))

# Attach clusters to DataFrame for visualization
plot_df = df.copy()
plot_df["KMeans_Cluster"] = kmeans_labels
plot_df["DBSCAN_Cluster"] = dbscan_labels
plot_df["Hierarchical_Cluster"] = hierarchical_labels

artifacts = {
    "scaler": scaler,
    "kmeans": kmeans,
    "metrics_df": metrics_df,
    "plot_df": plot_df
}

joblib.dump(artifacts, "clustering_suite.joblib")
print("\nExported clustering models to 'clustering_suite.joblib'")