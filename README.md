# 🤖 Machine Learning Implementation & Interactive Web Suite

A hands-on Machine Learning project implementing **11 core algorithms** across four foundational ML areas: **Regression**, **Classification**, **Clustering**, and **Dimensionality Reduction**.

Everything is built using Python, Scikit-Learn, and Pandas, paired with an interactive **Streamlit** dashboard that lets you test models with real-time sliders and visual charts.

---

## 📌 What is Inside This Project?

We cover 4 distinct machine learning pillars using real-world Kaggle datasets:

| Pillar | Problem Type | Dataset | Algorithms Implemented | Primary Evaluation Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **1. Regression** | Continuous Value Prediction | Ames Housing Dataset | Linear Regression (Ridge L2) | $R^2$ Score, RMSLE, MAE |
| **2. Classification** | Binary Outcome Prediction | Telco Customer Churn | Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| **3. Clustering** | Unsupervised Grouping | Mall Customer Segmentation | K-Means, DBSCAN, Hierarchical Clustering | Silhouette Score, Inertia |
| **4. Dimensionality Reduction** | High-to-Low Dimensional Mapping | Mall Customer Dataset | PCA (Linear), t-SNE (Non-Linear) | Explained Variance Ratio, KL-Divergence |

---

## 1. Regression: House Price Valuation

* **The Goal:** Predict the final selling price of a house based on its square footage, build quality, location, and rooms.
* **Why Linear Regression with Ridge?**
  * **Interpretability:** Unlike black-box models, Linear Regression gives us direct weights (coefficients) for each feature, showing exactly how much each square foot or bathroom adds to the property value.
  * **Ridge Penalty ($L_2$):** Features like basement area and above-ground living area are correlated. Ridge regularization prevents the model weights from exploding or overfitting.
* **How the Data is Preprocessed:**
  * **Log Transform:** House prices are right-skewed (a few mansions skew the average). We train on $\log(1 + \text{price})$ so the model treats percentage errors fairly across cheap and expensive homes.
  * **StandardScaler:** Centers numbers around $0$ so square footage (in thousands) doesn't overshadow bathroom count (single digits).
  * **One-Hot Encoding:** Converts text categories (like Neighborhoods) into binary $0$s and $1$s.
* **Validation Results:**
  * **$R^2$ Score:** `0.8720` (Explains ~87.2% of price variance)
  * **Validation MAE:** `~$18,635` average error on unseen test houses

---

## 2. Classification: Customer Churn Prediction

* **The Goal:** Predict whether a telecom customer will leave the service (`1 = Churn`) or stay (`0 = Retained`).
* **Why Compare 5 Models?** Different algorithms handle boundaries and decision splits in unique ways:
  1. **Logistic Regression:** Measures the probability of churn using a smooth S-shaped Sigmoid curve.
  2. **Decision Tree:** Splits data step-by-step based on yes/no questions (e.g., *Is contract month-to-month?*).
  3. **Random Forest:** Combines 150 different decision trees together to prevent overfitting.
  4. **Support Vector Machines (SVM):** Finds the widest mathematical street (margin) separating churners from non-churners using an RBF kernel.
  5. **Naive Bayes:** Uses basic probability and Bayes' Theorem to calculate churn likelihood.
* **Why Balanced Class Weights?** Only ~26% of customers churn in the dataset. Using `class_weight='balanced'` ensures the models don't ignore churners just to get high accuracy.
* **Validation Comparison Table:**
  * **Random Forest** achieved the strongest balance of **Accuracy (77.1%)** and **ROC-AUC (0.846)**.
  * **Logistic Regression** achieved the highest **Recall (79.7%)**, catching the most at-risk churners.

---

## 3. Clustering: Customer Segmentation

* **The Goal:** Group mall shoppers into behavioral segments based on Annual Income and Spending Score without having any pre-existing labels.
* **Algorithms Implemented:**
  1. **K-Means ($k=5$):** Places 5 center points (centroids) and assigns each customer to the nearest one.
  2. **DBSCAN:** Groups customers based on how densely packed together they are and flags isolated outliers as noise.
  3. **Hierarchical Clustering:** Builds a bottom-up tree (dendrogram) pairing similar customers until 5 distinct groups are formed.
* **How We Measure Quality:**
  * **Silhouette Score:** Measures how close a customer is to their own group versus neighboring groups (ranges from $-1$ to $+1$).
  * K-Means scored **`0.554`**, confirming 5 clearly separated groups (e.g., *High Earners / High Spenders*, *Careful Budgeters*, *Young Average Spenders*).

---

## 4. Dimensionality Reduction: PCA vs. t-SNE

* **The Goal:** Compress multi-dimensional customer data into a 2D plane so human eyes can view it on a graph.
* **Techniques Used:**
  * **PCA (Principal Component Analysis):** Draws new orthogonal axes through the data to capture maximum spread. It retains **`78.6%`** of total dataset variance in just 2 numbers.
  * **t-SNE:** A non-linear technique that preserves local clusters and neighborhoods, great for exploratory cluster visualization.

---

## 🛠️ Project File Structure

```text
├── train.csv                           # Ames Housing dataset (Regression)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Telco Churn dataset (Classification)
├── Mall_Customers.csv                  # Mall Customers dataset (Clustering & Dim Reduction)
│
├── train_model.py                      # Pipeline script for Linear Regression
├── train_classification_suite.py       # Pipeline script for all 5 Classifiers
├── train_clustering_suite.py           # Pipeline script for all 3 Clustering models
├── train_dim_reduction_suite.py        # Pipeline script for PCA and t-SNE
│
├── linear_house_price_pipeline.joblib  # Saved Linear Regression model artifact
├── classification_suite.joblib         # Saved Classification bundle
├── clustering_suite.joblib             # Saved Clustering bundle
├── dim_reduction_suite.joblib          # Saved PCA & t-SNE bundle
│
├── app.py                              # Master Streamlit Web App
├── requirements.txt                    # List of Python dependencies
└── README.md                           # Documentation
