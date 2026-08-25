# ==============================================================================
# 1. IMPORTING LIBRARIES
# WHAT: Imports tools for data manipulation, math, machine learning, and saving models.
# WHY:  Instead of coding math formulas and data loaders from scratch, we use 
#       battle-tested, industry-standard Python libraries.
# HOW:  - pandas: Reads and organizes CSV tables.
#       - numpy: Handles fast mathematical operations (like logarithms and exponentials).
#       - joblib: Exports/saves our trained Python objects into a single file.
#       - scikit-learn: Provides the algorithms, scalers, imputers, and evaluation metrics.
# ==============================================================================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error


# ==============================================================================
# 2. LOADING THE RAW DATASET
# WHAT: Reads the Kaggle Ames housing dataset from 'train.csv'.
# WHY:  The machine learning algorithm needs historical examples of houses 
#       (features) and their final selling prices (target) to learn patterns.
# HOW:  pd.read_csv() loads the CSV file into a 2D structured table (DataFrame).
# ==============================================================================
df = pd.read_csv("train.csv")


# ==============================================================================
# 3. FEATURE SELECTION & TARGET TRANSFORMATION
# WHAT: Picks 6 key numerical features, 3 key categorical features, and separates 
#       the target column ('SalePrice').
# WHY:  - Focusing on the highest-impact features prevents overfitting and keeps 
#         our web app UI clean and intuitive.
#       - Target prices are right-skewed (a few huge mansions inflate values).
# HOW:  - X contains the input attributes.
#       - np.log1p(df["SalePrice"]) applies log(1 + price) to compress extreme prices 
#         into a normal, bell-shaped distribution for linear regression.
# ==============================================================================
numeric_features = [
    "OverallQual",   # Overall material and finish quality (1 to 10)
    "GrLivArea",      # Above ground living area in square feet
    "TotalBsmtSF",    # Total square feet of basement area
    "GarageCars",     # Garage car capacity
    "FullBath",       # Full bathrooms above grade
    "YearBuilt"       # Original construction year
]

categorical_features = [
    "Neighborhood",   # Physical location within Ames city limits
    "ExterQual",      # Exterior material quality rating (Ex, Gd, TA, Fa)
    "KitchenQual"     # Kitchen quality rating (Ex, Gd, TA, Fa)
]

# Combine all input column names
features = numeric_features + categorical_features

# Separate inputs (X) and log-transformed target (y)
X = df[features]
y = np.log1p(df["SalePrice"])


# ==============================================================================
# 4. BUILDING THE PREPROCESSING PIPELINES
# WHAT: Prepares data so the mathematical Linear Regression formula can read it.
# WHY:  - Machine learning math cannot handle empty/null cells (NaN).
#       - Linear regression requires all features to be on a similar scale so 
#         large numbers (e.g., 2,000 sq ft) don't dominate small numbers (e.g., 2 baths).
#       - Computers cannot multiply text strings (e.g., "CollgCr")—they must be 
#         converted to 0s and 1s.
# HOW:  
#       1. numeric_transformer:
#          - SimpleImputer(median): Fills any missing number with the middle value.
#          - StandardScaler(): Scales numbers so Mean = 0 and Variance = 1.
#       2. categorical_transformer:
#          - SimpleImputer(most_frequent): Fills missing text with the most common category.
#          - OneHotEncoder(): Creates binary columns (1 if present, 0 if not).
#       3. ColumnTransformer:
#          - Directs numeric columns to the numeric pipeline and categorical columns 
#            to the categorical pipeline automatically.
# ==============================================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# ==============================================================================
# 5. ASSEMBLING THE FULL MACHINE LEARNING PIPELINE
# WHAT: Bundles preprocessing and the linear model into one continuous pipeline.
# WHY:  Guarantees that new data from the Streamlit UI undergoes the exact same 
#       transformations as the training data, preventing data leakage and bugs.
# HOW:  Uses Ridge(alpha=1.0) which is Linear Regression with L2 regularization 
#       to prevent correlated features from causing unstable weights.
# ==============================================================================
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0))
])


# ==============================================================================
# 6. TRAIN-TEST SPLIT & MODEL TRAINING
# WHAT: Splits data into 80% for training and 20% for testing, then fits the model.
# WHY:  Evaluating the model on data it has never seen before checks whether it 
#       can generalize to real-world homes.
# HOW:  - train_test_split(): Randomly partitions X and y.
#       - pipeline.fit(): Runs the preprocessors and solves the linear regression 
#         weights using least squares math.
# ==============================================================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the entire pipeline on the training subset
pipeline.fit(X_train, y_train)


# ==============================================================================
# 7. MODEL EVALUATION & METRICS
# WHAT: Calculates performance scores on the unseen 20% validation data.
# WHY:  Tells us how accurate our model is before deploying it to the web app.
# HOW:  
#       - R2 Score: Percentage of price variance explained (closer to 1.0 is better).
#       - RMSLE: Root Mean Squared Log Error (measures percentage error).
#       - MAE: Mean Absolute Error in real dollars after reversing the log scale (np.expm1).
# ==============================================================================
preds = pipeline.predict(X_val)

rmsle = root_mean_squared_error(y_val, preds)
r2 = r2_score(y_val, preds)
mae = mean_absolute_error(np.expm1(y_val), np.expm1(preds))

print("=== Model Training Summary ===")
print(f"Validation R2 Score: {r2:.4f} (Explains {r2*100:.1f}% of price variance)")
print(f"Validation RMSLE:    {rmsle:.4f}")
print(f"Validation MAE:      ${mae:,.2f} average dollar error")


# ==============================================================================
# 8. EXTRACTING LEARNED COEFFICIENTS (WEIGHTS)
# WHAT: Pulls out the mathematical weights (slopes) the model assigned to each feature.
# WHY:  Allows us to show the user inside Streamlit which features increase or decrease 
#       house value.
# HOW:  Retrieves one-hot encoded column names, pairs them with pipeline.named_steps["model"].coef_,
#       and saves them into a sorted DataFrame.
# ==============================================================================
preprocessor_fitted = pipeline.named_steps["preprocessor"]
cat_encoder = preprocessor_fitted.named_transformers_["cat"].named_steps["encoder"]
encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + encoded_cat_names

coefficients = pipeline.named_steps["model"].coef_

coef_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Coefficient (Weight)": coefficients
}).sort_values(by="Coefficient (Weight)", ascending=False)


# ==============================================================================
# 9. EXPORTING THE ARTIFACTS FOR STREAMLIT
# WHAT: Saves the trained pipeline, coefficient table, and dropdown lists to disk.
# WHY:  The Streamlit app (`app.py`) can simply load this pre-computed bundle 
#       instantly without recalculating anything.
# HOW:  joblib.dump() serializes the dictionary into 'linear_house_price_pipeline.joblib'.
# ==============================================================================
artifacts = {
    "pipeline": pipeline,
    "coef_df": coef_df,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "neighborhoods": sorted(df["Neighborhood"].dropna().unique().tolist()),
    "qualities": ["Ex", "Gd", "TA", "Fa", "Po"],
    "metrics": {
        "rmsle": rmsle,
        "r2": r2,
        "mae": mae
    }
}

joblib.dump(artifacts, "linear_house_price_pipeline.joblib")
print("Saved pipeline and artifacts to 'linear_house_price_pipeline.joblib'")