# 🏡 House Price Estimator (Linear Regression)

An interactive Machine Learning web application built with **Scikit-Learn** and **Streamlit** to predict residential property values using the Kaggle Ames Housing Dataset.

---

## 🚀 Key Highlights

- **Algorithm**: Regularized Linear Regression (Ridge L2 penalty)
- **Validation Metrics**: $R^2 = 0.8720$, $\text{RMSLE} = 0.1546$, $\text{MAE} \approx \$18,635$
- **Preprocessing Pipeline**:
  - `StandardScaler` on numerical features
  - `OneHotEncoder(handle_unknown='ignore')` on categorical features
  - Target logarithmic stabilization via $\log(1 + y)$ and inference recovery via $\exp(\hat{y}) - 1$
- **Frontend**: Streamlit interactive user interface

---

## 🛠️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone <YOUR-REPO-URL>
   cd <REPO-FOLDER>
   ```
