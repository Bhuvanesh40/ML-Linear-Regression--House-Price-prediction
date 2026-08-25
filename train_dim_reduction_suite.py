"""
# Dimensionality Reduction Suite:
# 1. PCA (Principal Component Analysis - Orthogonal linear variance maximization)
# 2. t-SNE (t-Distributed Stochastic Neighbor Embedding - Non-linear probabilistic manifold)
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. Load Data
df = pd.read_csv("Mall_Customers.csv")
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Algorithm 1: PCA (2 Components)
pca = PCA(n_components=2, random_state=42)
pca_transformed = pca.fit_transform(X_scaled)
pca_exp_var = pca.explained_variance_ratio_

# 3. Algorithm 2: t-SNE (2 Components)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_transformed = tsne.fit_transform(X_scaled)

# 4. Assemble 2D Projection DataFrame
dim_df = pd.DataFrame({
    "PCA_1": pca_transformed[:, 0],
    "PCA_2": pca_transformed[:, 1],
    "tSNE_1": tsne_transformed[:, 0],
    "tSNE_2": tsne_transformed[:, 1],
    "Age": df["Age"],
    "Annual Income (k$)": df["Annual Income (k$)"],
    "Spending Score (1-100)": df["Spending Score (1-100)"]
})

metrics_info = {
    "PCA_PC1_Var": pca_exp_var[0],
    "PCA_PC2_Var": pca_exp_var[1],
    "PCA_Total_Var": np.sum(pca_exp_var),
    "tSNE_KL_Divergence": tsne.kl_divergence_
}

print("=== Dimensionality Reduction Summary ===")
print(f"PCA Total Variance Retained: {metrics_info['PCA_Total_Var']*100:.2f}%")
print(f"t-SNE Final KL-Divergence:  {metrics_info['tSNE_KL_Divergence']:.4f}")

artifacts = {
    "dim_df": dim_df,
    "metrics_info": metrics_info,
    "scaler": scaler,
    "pca": pca
}

joblib.dump(artifacts, "dim_reduction_suite.joblib")
print("\nExported dimensionality reduction artifacts to 'dim_reduction_suite.joblib'")