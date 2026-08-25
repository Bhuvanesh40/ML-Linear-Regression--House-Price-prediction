import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import os

# ==============================================================================
# 1. APPLICATION CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Machine Learning Master Suite",
    page_icon="🤖",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_file_path(filename: str) -> str:
    """Safely locate model artifact either in root or in api/ directory."""
    paths_to_check = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "api", filename)
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, filename)

# ==============================================================================
# CACHED ARTIFACT LOADERS (Fast & Memory Efficient)
# ==============================================================================
@st.cache_resource
def load_regression_artifacts():
    return joblib.load(get_file_path("linear_house_price_pipeline.joblib"))

@st.cache_resource
def load_classification_artifacts():
    return joblib.load(get_file_path("classification_suite.joblib"))

@st.cache_resource
def load_clustering_artifacts():
    return joblib.load(get_file_path("clustering_suite.joblib"))

@st.cache_resource
def load_dim_reduction_artifacts():
    return joblib.load(get_file_path("dim_reduction_suite.joblib"))

# ==============================================================================
# 2. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🤖 ML Algorithms Suite")
pillar = st.sidebar.radio(
    "Select Machine Learning Pillar:",
    [
        "1. Regression (Linear / Ridge)",
        "2. Classification Suite (5 Models)",
        "3. Clustering Suite (3 Models)",
        "4. Dimensionality Reduction (PCA & t-SNE)"
    ]
)

# ==============================================================================
# PILLAR 1: REGRESSION (LINEAR REGRESSION / RIDGE)
# ==============================================================================
if pillar == "1. Regression (Linear / Ridge)":
    st.title("🏡 House Price Estimator (Linear Regression)")
    st.markdown("Predict property sales prices using **Regularized Linear Regression (Ridge)** on the Ames Housing dataset.")
    st.divider()

    try:
        artifacts = load_regression_artifacts()
        pipeline = artifacts["pipeline"]
        coef_df = artifacts["coef_df"]
        metrics = artifacts["metrics"]

        # Show performance metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Validation R² Score", f"{metrics['r2']:.4f}")
        m2.metric("Validation RMSLE", f"{metrics['rmsle']:.4f}")
        m3.metric("Validation MAE", f"${metrics['mae']:,.2f}")
        st.divider()

        st.subheader("Property Input Specifications")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Structure & Area**")
            overall_qual = st.slider("Overall Quality (1: Poor, 10: Excellent)", 1, 10, 7)
            gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1750, step=25)
            year_built = st.number_input("Year Built", 1880, 2026, 2005, step=1)

        with col2:
            st.markdown("**Basement & Amenities**")
            total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 4000, 950, step=25)
            garage_cars = st.selectbox("Garage Car Capacity", [0, 1, 2, 3, 4], index=2)
            full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=2)

        with col3:
            st.markdown("**Location & Finishes**")
            neighborhood = st.selectbox("Neighborhood", artifacts["neighborhoods"])
            exter_qual = st.selectbox("Exterior Material Quality", artifacts["qualities"], index=1)
            kitchen_qual = st.selectbox("Kitchen Quality", artifacts["qualities"], index=1)

        if st.button("Calculate Predicted Price (Linear Regression)", type="primary", use_container_width=True):
            input_df = pd.DataFrame([{
                "OverallQual": overall_qual, "GrLivArea": gr_liv_area, "TotalBsmtSF": total_bsmt_sf,
                "GarageCars": garage_cars, "FullBath": full_bath, "YearBuilt": year_built,
                "Neighborhood": neighborhood, "ExterQual": exter_qual, "KitchenQual": kitchen_qual
            }])
            
            log_pred = pipeline.predict(input_df)[0]
            final_price = float(np.expm1(log_pred))
            st.success(f"### Estimated Property Valuation: **${final_price:,.2f}**")

        with st.expander("🔍 View Learned Model Coefficients (Feature Weights)"):
            st.dataframe(coef_df, use_container_width=True, height=250)

    except Exception as e:
        st.error(f"Error loading regression artifacts: {e}")


# ==============================================================================
# PILLAR 2: CLASSIFICATION SUITE (5 ALGORITHMS)
# ==============================================================================
elif pillar == "2. Classification Suite (5 Models)":
    st.title("🎯 Classification Algorithms Suite")
    st.markdown("Predict customer churn probability across 5 benchmark classification models on the **Telco Dataset**.")
    st.divider()

    try:
        artifacts = load_classification_artifacts()
        pipelines = artifacts["pipelines"]
        metrics_df = artifacts["metrics_df"]

        selected_model = st.selectbox(
            "Choose Classification Algorithm:",
            ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machines", "Naive Bayes"]
        )

        st.subheader("Model Performance Comparison (Validation Set)")
        st.dataframe(metrics_df, use_container_width=True)
        st.divider()

        st.subheader(f"Inference using {selected_model}")
        c1, c2 = st.columns(2)

        with c1:
            tenure = st.slider("Tenure (Months with Provider)", 1, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 70.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", 18.0, 9000.0, 840.0, step=10.0)
            contract = st.selectbox("Contract Type", artifacts["contracts"])

        with c2:
            internet = st.selectbox("Internet Service", artifacts["internet_services"])
            payment = st.selectbox("Payment Method", artifacts["payment_methods"])
            security = st.selectbox("Online Security", artifacts["security_options"])
            tech = st.selectbox("Tech Support", artifacts["tech_options"])

        if st.button(f"Run Prediction ({selected_model})", type="primary", use_container_width=True):
            input_df = pd.DataFrame([{
                "tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
                "Contract": contract, "InternetService": internet, "PaymentMethod": payment,
                "OnlineSecurity": security, "TechSupport": tech
            }])
            
            active_pipeline = pipelines[selected_model]
            pred = int(active_pipeline.predict(input_df)[0])
            prob = float(active_pipeline.predict_proba(input_df)[0][1])

            if pred == 1:
                st.error(f"### Prediction: Churn Risk Detected (**Probability: {prob*100:.1f}%**)")
            else:
                st.success(f"### Prediction: Loyal Customer (**Probability of Staying: {(1-prob)*100:.1f}%**)")

    except Exception as e:
        st.error(f"Error loading classification artifacts: {e}")


# ==============================================================================
# PILLAR 3: CLUSTERING SUITE (3 ALGORITHMS)
# ==============================================================================
elif pillar == "3. Clustering Suite (3 Models)":
    st.title("🧩 Unsupervised Clustering Suite")
    st.markdown("Group customer segments on the **Mall Customers Dataset** without pre-existing labels.")
    st.divider()

    try:
        artifacts = load_clustering_artifacts()
        plot_df = artifacts["plot_df"]
        metrics_df = artifacts["metrics_df"]

        selected_cluster_algo = st.selectbox(
            "Choose Clustering Algorithm:",
            ["K-Means", "DBSCAN", "Hierarchical Clustering"]
        )

        st.subheader("Cluster Separation Metrics (Silhouette Scores)")
        st.dataframe(metrics_df, use_container_width=True)

        cluster_col_map = {
            "K-Means": "KMeans_Cluster",
            "DBSCAN": "DBSCAN_Cluster",
            "Hierarchical Clustering": "Hierarchical_Cluster"
        }
        active_col = cluster_col_map[selected_cluster_algo]

        st.subheader(f"Cluster Visualization: {selected_cluster_algo}")
        chart = alt.Chart(plot_df).mark_circle(size=70).encode(
            x=alt.X("Annual Income (k$):Q", title="Annual Income (k$)"),
            y=alt.Y("Spending Score (1-100):Q", title="Spending Score (1-100)"),
            color=alt.Color(f"{active_col}:N", scale=alt.Scale(scheme="tableau10"), title="Cluster ID"),
            tooltip=["Age", "Annual Income (k$)", "Spending Score (1-100)", active_col]
        ).properties(height=450).interactive()

        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading clustering artifacts: {e}")


# ==============================================================================
# PILLAR 4: DIMENSIONALITY REDUCTION (PCA & t-SNE)
# ==============================================================================
elif pillar == "4. Dimensionality Reduction (PCA & t-SNE)":
    st.title("📉 Dimensionality Reduction Suite")
    st.markdown("Compress multi-dimensional customer feature space into 2D projections using **PCA** and **t-SNE**.")
    st.divider()

    try:
        artifacts = load_dim_reduction_artifacts()
        dim_df = artifacts["dim_df"]
        metrics = artifacts["metrics_info"]

        selected_dim_algo = st.selectbox(
            "Choose Dimensionality Reduction Technique:",
            ["PCA (Principal Component Analysis)", "t-SNE (t-Distributed Stochastic Neighbor Embedding)"]
        )

        if selected_dim_algo == "PCA (Principal Component Analysis)":
            total_var = float(metrics['PCA_Total_Var']) * 100
            pc1_var = float(metrics['PCA_PC1_Var']) * 100
            pc2_var = float(metrics['PCA_PC2_Var']) * 100
            st.info(f"**Total Information Retained in 2D**: `{total_var:.2f}%` (PC1: `{pc1_var:.1f}%`, PC2: `{pc2_var:.1f}%`)")
            x_axis, y_axis = "PCA_1", "PCA_2"
        else:
            kl_div = float(metrics['tSNE_KL_Divergence'])
            st.info(f"**Non-linear Neighborhood Embedding** — Final KL Divergence: `{kl_div:.4f}`")
            x_axis, y_axis = "tSNE_1", "tSNE_2"

        chart = alt.Chart(dim_df).mark_circle(size=70).encode(
            x=alt.X(f"{x_axis}:Q", title=f"{x_axis}"),
            y=alt.Y(f"{y_axis}:Q", title=f"{y_axis}"),
            color=alt.Color("Spending Score (1-100):Q", scale=alt.Scale(scheme="viridis"), title="Spending Score"),
            tooltip=["Age", "Annual Income (k$)", "Spending Score (1-100)"]
        ).properties(height=450).interactive()

        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading dimensionality reduction artifacts: {e}")