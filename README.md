# IBM-internship-Project-

# Estimation of Obesity Levels Based on Eating Habits and Physical Condition

An end-to-end Machine Learning and Deep Learning pipeline to predict and classify individual obesity levels using dietary patterns, physical habits, and demographic attributes.

---

## 📌 Project Overview
Obesity is a major global health risk linked to diabetes, hypertension, and cardiovascular diseases. This project builds a machine learning pipeline that evaluates lifestyle and physiological attributes to classify individuals into specific clinical weight tiers (multiclass) as well as binary obesity diagnoses (Obese vs. Not Obese)[cite: 1, 2, 4].

The project implements data preprocessing via Scikit-Learn `ColumnTransformer`, hyperparameter optimization through `GridSearchCV` with Stratified K-Fold cross-validation, and comparative modeling with **Logistic Regression**, **Random Forest Classifier**, and an **Artificial Neural Network (ANN)** built with TensorFlow/Keras[cite: 1, 2, 4].

---

## 📊 Dataset Details
* **Source**: [UCI Machine Learning Repository - Estimation of Obesity Levels Based On Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
* **Dataset Shape**: 2,111 instances, 17 attributes
* **Target Variable (`NObeyesdad`)**:
  * `Insufficient_Weight`[cite: 2, 4]
  * `Normal_Weight`[cite: 2, 4]
  * `Overweight_Level_I`[cite: 2, 4]
  * `Overweight_Level_II`[cite: 2, 4]
  * `Obesity_Type_I`[cite: 2, 4]
  * `Obesity_Type_II`[cite: 2, 4]
  * `Obesity_Type_III`[cite: 2, 4]

### Key Feature Breakdown
* **Demographic & Physiological**: `Gender`, `Age`, `Height`, `Weight`, `family_history_with_overweight`
* **Dietary Patterns**: `FAVC` (Frequent consumption of high caloric food), `FCVC` (Frequency of consumption of vegetables), `NCP` (Number of main meals), `CAEC` (Consumption of food between meals), `CH2O` (Daily water consumption), `CALC` (Consumption of alcohol)
* **Physical Condition & Habits**: `SCC` (Calories consumption monitoring), `FAF` (Physical activity frequency), `TUE` (Time using technology devices), `SMOKE`, `MTRANS` (Mode of transportation)

---

## ⚙️ Data Preprocessing Pipeline
* **Missing Values**: Verified complete integrity (0 null/missing values).
* **Target Encoding**:
  * **Multiclass Models**: Encoded 7 categorical levels to integer labels (`0` to `6`) using `LabelEncoder`[cite: 2, 4].
  * **ANN Binary Model**: Binarized into `1` (Obese) and `0` (Not Obese)[cite: 1].
* **Feature Transformation (`ColumnTransformer`)**:
  * **Numerical Features**: Scaled to zero mean and unit variance using `StandardScaler`[cite: 1, 2, 4].
  * **Categorical Features**: Transformed via `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`[cite: 1, 2, 4].
* **Leakage Prevention**: Transformations and classifiers were encapsulated within `sklearn.pipeline.Pipeline` workflows during 5-fold cross-validation[cite: 2, 4].

---

## 🧠 Model Architectures & Results

### Multiclass Classification (7 Classes)
| Model | Optimization / Configuration | Test Accuracy | Macro F1-Score |
| :--- | :--- | :---: | :---: |
| **Logistic Regression** | `lbfgs` solver, $C=100$, 5-Fold CV[cite: 2] | **95.98%**[cite: 2] | **0.96**[cite: 2] |
| **Random Forest** | 200 Trees, `min_samples_split=5`, 5-Fold Stratified CV[cite: 4] | **93.62%**[cite: 4] | **0.94**[cite: 4] |

### Binary Classification (Obese vs. Not Obese)
| Model | Architecture & Settings | Test Accuracy | Test Loss |
| :--- | :--- | :---: | :---: |
| **Artificial Neural Network (ANN)** | Dense(64, ReLU) $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(32, ReLU) $\rightarrow$ Dense(1, Sigmoid) with Early Stopping[cite: 1] | **98.58%**[cite: 1] | **0.0323**[cite: 1] |

---

## 📈 Key Insights & Feature Importance
* **Dominant Risk Factors**: `Weight` was the most critical factor (29.34% importance weight in Random Forest), followed by `Age` (8.97%), `FCVC` (Vegetable intake frequency, 8.38%), and `Height` (8.04%)[cite: 4].
* **Severe Obesity Detection**: Both Random Forest and Logistic Regression achieved over **0.98–1.00 recall and precision** on high-risk categories (`Obesity_Type_II` and `Obesity_Type_III`), minimizing false negatives[cite: 2, 4].

---

## 📁 Repository Structure
```text
├── ObesityDataSet_raw_and_data_sinthetic.csv   # UCI Dataset
├── ann.ipynb                                    # ANN Deep Learning implementation (Binary Task)
├── logisticsRegression.ipynb                   # Optimized Logistic Regression pipeline (Multiclass)
├── RandomForest.ipynb                # Random Forest pipeline & feature importance (Multiclass)
└── README.md                                   # Project documentation
