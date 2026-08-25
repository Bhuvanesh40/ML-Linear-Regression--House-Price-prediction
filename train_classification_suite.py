"""
# Classification Algorithms Suite:
# 1. Logistic Regression (Probabilistic Sigmoid classification)
# 2. Decision Tree (Entropy/Gini recursive information gain splits)
# 3. Random Forest (Bootstrap aggregation ensemble of decision trees)
# 4. Support Vector Machines - SVM (Maximum margin hyperplanes with RBF kernel)
# 5. Gaussian Naive Bayes (Bayesian conditional probability with Gaussian priors)
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import the 5 Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 1. Load and clean Telco Churn Data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

y = df["Churn"].map({"Yes": 1, "No": 0})
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport"]

X = df[numeric_features + categorical_features]

# 2. Preprocessing Transformers
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, numeric_features),
    ("cat", cat_transformer, categorical_features)
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Define the 5 Models
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42),
    "Support Vector Machines": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "Naive Bayes": GaussianNB()
}

trained_pipelines = {}
metrics_table = []

# 4. Train and Evaluate each algorithm
for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

    preds = pipe.predict(X_val)
    probs = pipe.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds)
    rec = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    auc = roc_auc_score(y_val, probs)

    metrics_table.append({
        "Algorithm": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

metrics_df = pd.DataFrame(metrics_table)
print("=== Classification Models Evaluation ===")
print(metrics_df.to_string(index=False))

# 5. Export serialized bundle
artifacts = {
    "pipelines": trained_pipelines,
    "metrics_df": metrics_df,
    "contracts": sorted(df["Contract"].dropna().unique().tolist()),
    "internet_services": sorted(df["InternetService"].dropna().unique().tolist()),
    "payment_methods": sorted(df["PaymentMethod"].dropna().unique().tolist()),
    "security_options": sorted(df["OnlineSecurity"].dropna().unique().tolist()),
    "tech_options": sorted(df["TechSupport"].dropna().unique().tolist())
}

joblib.dump(artifacts, "classification_suite.joblib")
print("\nExported classification models to 'classification_suite.joblib'")