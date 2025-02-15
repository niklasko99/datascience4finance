# DataScience4Finance

## 📌 Project Overview
This project aims to **predict future earnings changes using financial data** from publicly traded US companies. Leveraging **machine learning techniques** such as **Random Forest, XGBoost, and Logistic Regression**, the goal is to improve earnings forecasting, a key challenge in financial analysis. The dataset is sourced from **[Kaggle](https://www.kaggle.com/datasets/vadimvanak/step-2)** and contains financial statement data from **SEC filings** from US stock listed companies.

This project was developed as part of the **"Data Science in Finance"** course at the **Technical University of Munich (TUM)**.

---

## 🛠️ Installation & Setup

1. Create a new Conda environment:
```bash
conda create --name <your_environment_name> python=3.11
```


2. Activate the environment:
```bash
conda activate <your_environment_name>
```

3. Navigate to the project directory:
```bash
cd path/to/your/project/folder
```

4. Install dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

## 📂 Project Structure

The repository is structured into different folders for exploratory analysis, feature selection, model training, and output storage.

```
📦 DataScience4Finance
 ┣ 📂 explanatory_analysis
 ┃ ┗ 📜 initial_analysis.ipynb
 ┣ 📂 data
 ┃ ┗ 📜 (Raw and processed data files + external data) 
 ┣ 📂 feature_selection
 ┃ ┣ 📂 1_iteration
 ┃ ┃ ┣ 📂 rf
 ┃ ┃ ┃ ┣ 📜 rf_feature_importances.csv
 ┃ ┃ ┃ ┣ 📜 rf_model.pkl
 ┃ ┃ ┃ ┣ 📜 rf_roc_curve.png
 ┃ ┃ ┃ ┣ 📜 rf_search_results.csv
 ┃ ┃ ┃ ┣ 📜 rf_test_predictions.csv
 ┃ ┃ ┃ ┗ 📜 rf_train_results.csv
 ┃ ┃ ┣ 📂 xgb (same output files as for random forest)
 ┃ ┃ ┗ 📜 mean_feature_importance_1_iteration.png
 ┃ ┣ 📂 2_iteration (same structure as previous folder)
 ┃ ┣ 📂 3_iteration (same structure as previous folder)
 ┃ ┣ 📜 feature_selection.py
 ┃ ┣ 📜 rf_importance_kept_features.png
 ┃ ┗ 📜 xgb_importance_kept_features.png
 ┣ 📂 output
 ┃ ┣ 📂 logistic_regression (same output files as for rf)
 ┃ ┣ 📂 rf
 ┃ ┃ ┣ 📜 rf_best_hyperparameters.json
 ┃ ┃ ┣ 📜 rf_feature_importances.csv
 ┃ ┃ ┣ 📜 rf_feature_importances.png
 ┃ ┃ ┣ 📜 rf_model.pkl
 ┃ ┃ ┣ 📜 rf_roc_curve.png
 ┃ ┃ ┣ 📜 rf_search_results.csv
 ┃ ┃ ┣ 📜 rf_search_results1.png
 ┃ ┃ ┣ 📜 rf_test_predictions.csv
 ┃ ┃ ┣ 📜 rf_train_predictions.csv
 ┃ ┃ ┗ 📜 rf_train_results.csv
 ┃ ┣ 📂 xgb (same output files as for rf)
 ┃ ┗ 📂 xgb_robustness_check (same output files as for rf)
 ┣ 📂 training
 ┃ ┣ 📜 final_evaluation.ipynb
 ┃ ┣ 📜 helper_functions.py
 ┃ ┣ 📜 logistic_regression.py
 ┃ ┣ 📜 rf.py
 ┃ ┣ 📜 xgb.py
 ┃ ┗ 📜 xgb_robustness_check.py
 ┣ 📜 .gitignore
 ┣ 📜 preprocessing.py
 ┣ 📜 README.md
 ┗ 📜 requirements.txt
 ```

---

## 📖 Repository Explanation

Here’s what each folder contains:

### 📂 `explanatory_analysis/`
- Contains **initial_analysis.ipynb**, which performs **exploratory data analysis (EDA)** on the dataset.
- **First step after retrieving the dataset from Kaggle**:  
  - Run `initial_analysis.ipynb` to **filter the dataset** (e.g., only **10-K reports** are considered).  
  - The **first preprocessed dataset** is then stored in the **`data/` folder** for further use.
  - Run `preprocessing.py` to store cleaned and processed files.


### 📂 `data/`
- Stores the **raw and processed datasets**.
- Some large files are **ignored via `.gitignore`**, but datasets can be downloaded from **[Kaggle](https://www.kaggle.com/datasets/vadimvanak/step-2)**. 

### 📂 `feature_selection/`
- This folder contains scripts and outputs related to **recursive feature elimination (RFE)**.
- **Subfolders (`1_iteration`, `2_iteration`, `3_iteration`)** store different iterations of feature selection.
- The `rf/` and `xgb/` subfolders contain **feature importance scores** and model-specific outputs.

### 📂 `output/`
- Stores **model results**, including:
  - **Hyperparameter settings**
  - **Feature importance rankings**
  - **Model evaluation metrics**
  - **Trained models (`.pkl` files)**

### 📂 `training/`
- **Contains scripts for training models:**
  - `logistic_regression.py` → Logistic Regression Model
  - `rf.py` → Random Forest Model
  - `xgb.py` → XGBoost Model
  - `xgb_robustness_check.py` → XGB model without imputed data
  - `final_evaluation.ipynb` → **Final comparison of models**.

### 📜 Key Files in the Main Directory
- **`.gitignore`** → Prevents large files from being committed.
- **`preprocessing.py`** → Runs data preprocessing and creates a **cleaned dataset**.
- **`requirements.txt`** → Lists all required dependencies for the project.
- **`README.md`** → This documentation file.

## 📈 Model Performance
The models are evaluated based on **Area Under the Curve (AUC)** and other classification metrics.

| Model             | AUC (%)  | Accuracy (%) | Precision (%) | Recall (%) |
|------------------|---------|-------------|--------------|------------|
| **XGBoost**      | **68.10** | **63.27**    | **63.61**     | 53.96      |
| **Random Forest** | 66.22    | 60.45       | 57.17        | 68.46      |
| Logistic Reg.    | 60.31    | 54.42       | 51.35        | **86.58**  |

- **XGBoost** performs best in terms of overall predictive power (AUC: **68.10%**).
- **Random Forest** provides more balanced predictions.
- **Logistic Regression** has the highest recall but struggles with precision.

---

## 📌 Key Features Driving Earnings Predictions
The most influential features for predicting earnings changes include:

**📊 Financial Metrics:**
- **Earnings per Share (EPS)**
- **Comprehensive Income**
- **Profit/Loss Change**
- **Liabilities Change**

**📉 Macroeconomic Indicators:**
- **U.S. Business Confidence (Lagged & Change)**
- **G20 Business Confidence (Lagged)**

**📅 Temporal Variables:**
- **Pre-COVID period**
- **Time Elapsed Since 2013**

These features impact model performance and help capture financial trends.


---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🔗 References
This project builds on financial research, particularly:
- Chen et al. (2022) - Machine Learning for Earnings Prediction
- Freeman et al. (1982) - Financial Ratios in Forecasting
- Shroff et al. (2014) - Predicting Corporate Earnings Trends

---

## 📧 Contact
For any questions or collaborations, feel free to reach out:
- **Authors:** Berivan Kevser Yatki, Dogukan Dogu, Niklas Kothe
- **Affiliation:** TUM - Technical University of Munich