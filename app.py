import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# 1. PAGE SETUP
# WHAT: Sets up the browser tab title, favicon, and screen width.
# WHY:  Makes the web page look clean and utilize the full monitor width.
# HOW:  Calls st.set_page_config() before any other Streamlit commands.
# ---------------------------------------------------------
st.set_page_config(
    page_title="House Price Estimator",
    page_icon="🏡",
    layout="wide"
)

# ---------------------------------------------------------
# 2. LOAD MODEL ARTIFACTS
# WHAT: Loads the saved Scikit-Learn pipeline and metadata from disk.
# WHY:  We avoid training the model from scratch every time the user clicks a button.
# HOW:  Uses @st.cache_resource so Python only loads the file once into memory.
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Read the serialized joblib file containing the pipeline and categorical lists
    return joblib.load("linear_house_price_pipeline.joblib")

# Unpack the dictionary into variables we need for the UI and inference
artifacts = load_artifacts()
pipeline = artifacts["pipeline"]            # Trained Pipeline (Preprocessor + Ridge Model)
coef_df = artifacts["coef_df"]              # Feature weights table
neighborhoods = artifacts["neighborhoods"]  # List of unique neighborhoods
qualities = artifacts["qualities"]          # List of quality ratings ('Ex', 'Gd', etc.)

# ---------------------------------------------------------
# 3. PAGE HEADER
# WHAT: Displays the main title of the application.
# WHY:  Gives the interface a clean, straightforward headline.
# ---------------------------------------------------------
st.title("House Price Estimator")
st.divider()

# ---------------------------------------------------------
# 4. USER INPUT CONTROLS
# WHAT: Creates interactive sliders, number boxes, and dropdown menus.
# WHY:  Allows the user to input specific property characteristics.
# HOW:  Uses st.columns(3) to create a clean, responsive 3-column layout.
# ---------------------------------------------------------
st.subheader("Property Input Specifications")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Property Structure**")
    # Slider for general quality score (1 to 10)
    overall_qual = st.slider("Overall Quality (1: Poor, 10: Excellent)", min_value=1, max_value=10, value=7)
    # Number input for square footage above ground
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=6000, value=1750, step=25)
    # Number input for construction year
    year_built = st.number_input("Year Built", min_value=1880, max_value=2026, value=2005, step=1)

with col2:
    st.markdown("**Basement & Amenities**")
    # Number input for basement square footage
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=4000, value=950, step=25)
    # Dropdown for garage capacity
    garage_cars = st.selectbox("Garage Car Capacity", options=[0, 1, 2, 3, 4], index=2)
    # Dropdown for full bathroom count
    full_bath = st.selectbox("Full Bathrooms", options=[1, 2, 3, 4], index=2)

with col3:
    st.markdown("**Location & Finishes**")
    # Dropdown for neighborhood name
    neighborhood = st.selectbox("Neighborhood", options=neighborhoods)
    # Dropdown for exterior material quality
    exter_qual = st.selectbox("Exterior Material Quality", options=qualities, index=1,
                              help="Ex: Excellent, Gd: Good, TA: Typical/Average, Fa: Fair, Po: Poor")
    # Dropdown for kitchen quality
    kitchen_qual = st.selectbox("Kitchen Quality", options=qualities, index=1,
                               help="Ex: Excellent, Gd: Good, TA: Typical/Average, Fa: Fair, Po: Poor")

st.divider()

# ---------------------------------------------------------
# 5. PREDICTION & INFERENCE ENGINE
# WHAT: Captures all user inputs, formats them into a DataFrame, runs the pipeline, and displays the price.
# WHY:  Connects user inputs to the mathematical linear regression model.
# HOW:  
#       1. Collect inputs into a 1-row Pandas DataFrame.
#       2. Pass through pipeline: Imputes missing data -> Scales numbers -> One-hot encodes text.
#       3. Model calculates log-predicted price via: y_hat = w^T * X + b.
#       4. Reverse the log transformation using np.expm1(log_pred) to get real USD ($).
# ---------------------------------------------------------
if st.button("Calculate Predicted Price", type="primary", use_container_width=True):
    # Step 5a: Structure the inputs into a single-row DataFrame matching the training schema
    input_data = pd.DataFrame([{
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "GarageCars": garage_cars,
        "FullBath": full_bath,
        "YearBuilt": year_built,
        "Neighborhood": neighborhood,
        "ExterQual": exter_qual,
        "KitchenQual": kitchen_qual
    }])
    
    # Step 5b: Run the pipeline to get the log-transformed prediction
    log_pred = pipeline.predict(input_data)[0]
    
    # Step 5c: Reverse log1p using expm1 (exponential minus 1) to retrieve the actual dollar value
    final_price = np.expm1(log_pred)
    
    # Step 5d: Display the final prediction in a green success banner formatted as currency
    st.success(f"### Estimated Valuation: **${final_price:,.2f}**")

# ---------------------------------------------------------
# 6. MODEL INTERPRETABILITY (COEFFICIENT BREAKDOWN)
# WHAT: Displays a table of the learned weights (coefficients) from the Linear Model.
# WHY:  Linear regression is uniquely explainable; users can inspect which factors raise or lower value.
# HOW:  Renders the pre-computed coef_df DataFrame inside an expandable drawer.
# ---------------------------------------------------------
with st.expander("🔍 View Feature Impact (Learned Coefficients)"):
    st.markdown("Positive coefficients increase property value; negative coefficients reduce it:")
    st.dataframe(coef_df, use_container_width=True, height=250)